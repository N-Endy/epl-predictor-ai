using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using Microsoft.Extensions.Caching.Memory;

namespace PredictorBlazor.Services;

public record FixturePrediction(
    string HomeTeam,
    string AwayTeam,
    DateTime Kickoff,
    int Round,
    string PredictedOutcome,
    float Probability,
    float HomeWinProb = 0f,
    float DrawProb = 0f,
    float AwayWinProb = 0f,
    float ExpectedHomeGoals = 0f,
    float ExpectedAwayGoals = 0f,
    string? TacticalInsight = null
)
{
    public string GenerateTacticalSummary()
    {
        var outcome = PredictedOutcome switch
        {
            "H" => $"{HomeTeam} Win",
            "A" => $"{AwayTeam} Win",
            _ => "Draw"
        };

        var xgDiff = ExpectedHomeGoals - ExpectedAwayGoals;
        var favorite = xgDiff >= 0 ? HomeTeam : AwayTeam;
        var underdog = xgDiff >= 0 ? AwayTeam : HomeTeam;
        var absXgDiff = Math.Abs(xgDiff);

        if (PredictedOutcome == "D")
        {
            return $"Evenly matched tactical contest. Expected goals are tightly aligned ({ExpectedHomeGoals:0.0} vs {ExpectedAwayGoals:0.0}), pointing to a high probability of points being shared.";
        }
        else if (Probability > 0.65f)
        {
            return $"Strong {outcome} projected ({Probability:P0} confidence). {favorite} holds a commanding +{absXgDiff:0.1} expected goals advantage and significant Elo strength over {underdog}.";
        }
        else
        {
            return $"Competitive matchup leaning towards a {outcome} ({Probability:P0} confidence), with expected goals projected at {ExpectedHomeGoals:0.0}–{ExpectedAwayGoals:0.0}.";
        }
    }
};

public record BacktestItem(
    string HomeTeam,
    string AwayTeam,
    DateTime Kickoff,
    string Result,
    string ActualOutcome,
    string PredictedOutcome,
    float Probability,
    float HomeWinProb = 0f,
    float DrawProb = 0f,
    float AwayWinProb = 0f
);

public record BacktestResult(
    int Round,
    int Correct,
    int Total,
    double Accuracy,
    List<BacktestItem> Items
);

public class PredictionSession
{
    public int LastCompletedRound { get; init; }
    public int NextRound { get; init; }
    public List<FixturePrediction> NextRoundPredictions { get; init; } = new();
    public BacktestResult? Backtest { get; init; }
    public string InfoMessage { get; init; } = string.Empty;
    public List<string> LoadedSeasons { get; init; } = new();
    public int TotalMatchesTrained { get; init; }
}

public class PredictorService
{
    private readonly ILogger<PredictorService> _log;
    private readonly IConfiguration _config;
    private readonly IMemoryCache _cache;
    private readonly object _seasonLock = new();
    private int? _resolvedSeasonYear;
    private const string DefaultCsvFileName = "epl.csv";
    private const int HistoricalSeasonLookback = 8;

    public PredictorService(ILogger<PredictorService> log, IConfiguration config, IMemoryCache cache)
    {
        _log = log;
        _config = config;
        _cache = cache;
    }

    /// <summary>
    /// EPL seasons start in August; Jan–Jun still belongs to the previous season year.
    /// July is treated as the new season year so the resolver can probe upstream early.
    /// </summary>
    public static int DeriveSeasonYear(DateTime now) => now.Month >= 7 ? now.Year : now.Year - 1;

    public int? GetLocalCsvSeasonYear(string wwwrootPath)
    {
        var filePath = Path.Combine(wwwrootPath, DefaultCsvFileName);
        if (!File.Exists(filePath))
            return null;
        return GetSeasonYearFromMatches(LoadMatches(filePath));
    }

    public async Task<bool> IsLocalSeasonBehindAsync(string wwwrootPath, CancellationToken ct = default)
    {
        var local = GetLocalCsvSeasonYear(wwwrootPath);
        var current = await ResolveCurrentSeasonYearAsync(ct);
        return local is null || local.Value < current;
    }

    public string GetHarvestedSeasonRangeLabel(string wwwrootPath)
    {
        var years = new List<int>();
        var local = GetLocalCsvSeasonYear(wwwrootPath);
        if (local.HasValue)
            years.Add(local.Value);

        if (Directory.Exists(wwwrootPath))
        {
            foreach (var f in Directory.GetFiles(wwwrootPath, "epl-*.csv"))
            {
                var name = Path.GetFileNameWithoutExtension(f);
                if (name.StartsWith("epl-", StringComparison.OrdinalIgnoreCase)
                    && int.TryParse(name.AsSpan(4), out var y))
                {
                    years.Add(y);
                }
            }
        }

        if (years.Count == 0)
            return "historical seasons from fixturedownload.com";

        return $"{years.Min()}–{years.Max()}";
    }

    public async Task<int> ResolveCurrentSeasonYearAsync(CancellationToken ct = default)
    {
        lock (_seasonLock)
        {
            if (_resolvedSeasonYear.HasValue)
                return _resolvedSeasonYear.Value;
        }

        var overrideYear = _config.GetValue<int?>("Epl:SeasonYear");
        if (overrideYear is > 2000)
        {
            lock (_seasonLock)
                _resolvedSeasonYear = overrideYear.Value;
            _log.LogInformation("Using configured Epl:SeasonYear={Year}", overrideYear.Value);
            return overrideYear.Value;
        }

        var derived = DeriveSeasonYear(DateTime.UtcNow);
        using var client = CreateHttpClient(15);
        int resolved;
        if (await SeasonCsvExistsAsync(client, derived, ct))
        {
            resolved = derived;
        }
        else
        {
            resolved = derived - 1;
            _log.LogWarning("Season {Year} CSV not found upstream; falling back to {Fallback}", derived, resolved);
        }

        lock (_seasonLock)
            _resolvedSeasonYear = resolved;
        return resolved;
    }

    public class MatchRaw
    {
        public int MatchNumber { get; set; }
        public int RoundNumber { get; set; }
        public DateTime Date { get; set; }
        public string HomeTeam { get; set; } = "";
        public string AwayTeam { get; set; } = "";
        public string Location { get; set; } = "";
        public string? Result { get; set; }
    }

    public async Task<PredictionSession> PredictAsync(Stream csvStream, int backtestRound = 0, CancellationToken ct = default)
    {
        var sw = Stopwatch.StartNew();
        _log.LogInformation("PredictAsync: starting. backtestRound={BacktestRound}", backtestRound);
        using var reader = new StreamReader(csvStream, leaveOpen: true);
        var csvText = await reader.ReadToEndAsync(ct);
        var raw = LoadMatchesFromText(csvText);
        var result = await PredictInternalAsync(raw, backtestRound: backtestRound);
        sw.Stop();
        _log.LogInformation("PredictAsync finished in {ElapsedMs} ms. Last={Last}, Next={Next}, Predictions={Count}",
            sw.ElapsedMilliseconds, result.LastCompletedRound, result.NextRound, result.NextRoundPredictions.Count);
        return result;
    }

    public async Task<PredictionSession> PredictFromFileIfExistsAsync(
        string wwwrootPath, 
        int backtestRound = 0, 
        int endBacktestRound = 0, 
        bool forceRefresh = false)
    {
        var sw = Stopwatch.StartNew();
        var filePath = Path.Combine(wwwrootPath, DefaultCsvFileName);
        _log.LogInformation("PredictFromFileIfExistsAsync: checking {Path}", filePath);
        if (!File.Exists(filePath))
        {
            _log.LogWarning("File not found at {Path}", filePath);
            return new PredictionSession { InfoMessage = $"No {DefaultCsvFileName} found under wwwroot." };
        }

        var fileInfo = new FileInfo(filePath);
        var cacheKey = $"pred_session_{fileInfo.LastWriteTimeUtc.Ticks}_{backtestRound}_{endBacktestRound}";
        if (!forceRefresh && _cache.TryGetValue(cacheKey, out PredictionSession? cached) && cached is not null)
        {
            sw.Stop();
            _log.LogInformation("PredictFromFileIfExistsAsync: served from memory cache in {ElapsedMs} ms.", sw.ElapsedMilliseconds);
            return cached;
        }

        var raw = LoadMatches(filePath);
        var result = await PredictInternalAsync(raw, backtestRound: backtestRound, endBacktestRound: endBacktestRound, wwwrootPath: wwwrootPath);
        sw.Stop();
        _log.LogInformation("PredictFromFileIfExistsAsync finished in {ElapsedMs} ms.", sw.ElapsedMilliseconds);

        if (result.NextRoundPredictions.Count > 0 || result.Backtest is not null)
        {
            _cache.Set(cacheKey, result, TimeSpan.FromHours(6));
        }

        return result;
    }

    public async Task<string> UpdateFixturesAsync(string wwwrootPath, CancellationToken ct = default)
    {
        using var client = CreateHttpClient(25);
        var seasonYear = await ResolveCurrentSeasonYearAsync(ct);
        var url = FixtureDownloadUrl(seasonYear);
        var filePath = Path.Combine(wwwrootPath, DefaultCsvFileName);
        _log.LogInformation("UpdateFixturesAsync: downloading from {Url} to {Path}", url, filePath);

        var csvContent = await client.GetStringAsync(url, ct);

        if (!LooksLikeFixtureCsv(csvContent))
            throw new Exception("Downloaded CSV content is empty or invalid.");

        ArchiveCurrentSeasonIfRolled(wwwrootPath, seasonYear);

        await File.WriteAllTextAsync(filePath, csvContent, ct);
        var lineCount = csvContent.Split('\n', StringSplitOptions.RemoveEmptyEntries).Length - 1;
        _log.LogInformation("UpdateFixturesAsync: successfully updated {Path} ({Lines} matches) for {Year}/{Next}",
            filePath, lineCount, seasonYear, seasonYear + 1);
        return $"Updated {seasonYear}/{seasonYear + 1} fixtures ({lineCount} matches).";
    }

    public async Task<string> UpdateAllFixturesAndSeasonsAsync(string wwwrootPath, CancellationToken ct = default)
    {
        using var client = CreateHttpClient(30);

        int updatedCount = 0;
        var summary = new List<string>();
        var seasonYear = await ResolveCurrentSeasonYearAsync(ct);

        try
        {
            var currContent = await client.GetStringAsync(FixtureDownloadUrl(seasonYear), ct);
            if (LooksLikeFixtureCsv(currContent))
            {
                ArchiveCurrentSeasonIfRolled(wwwrootPath, seasonYear);
                var currPath = Path.Combine(wwwrootPath, DefaultCsvFileName);
                await File.WriteAllTextAsync(currPath, currContent, ct);
                updatedCount++;
                summary.Add($"{seasonYear}/{seasonYear + 1} (Current)");
            }
        }
        catch (Exception ex)
        {
            _log.LogWarning(ex, "Failed to download current season from primary URL");
        }

        for (int year = seasonYear - 1; year >= seasonYear - HistoricalSeasonLookback; year--)
        {
            try
            {
                var candidateUrls = new[]
                {
                    FixtureDownloadUrl(year),
                    $"https://fixturedownload.com/download/csv/epl-{year}"
                };

                string? csvContent = null;
                foreach (var url in candidateUrls)
                {
                    try
                    {
                        var res = await client.GetStringAsync(url, ct);
                        if (LooksLikeFixtureCsv(res))
                        {
                            csvContent = res;
                            break;
                        }
                    }
                    catch { }
                }

                if (!string.IsNullOrWhiteSpace(csvContent))
                {
                    var seasonPath = Path.Combine(wwwrootPath, $"epl-{year}.csv");
                    await File.WriteAllTextAsync(seasonPath, csvContent, ct);
                    updatedCount++;
                    summary.Add($"{year}/{year + 1}");
                    _log.LogInformation("Successfully harvested season {Year}", year);
                }
            }
            catch (Exception ex)
            {
                _log.LogWarning(ex, "Failed to download historical season {Year}", year);
            }
        }

        return $"Successfully synced {updatedCount} seasons: {string.Join(", ", summary)}";
    }

    public List<string> GetAvailableSeasons(string wwwrootPath)
    {
        var result = new List<string>();
        if (File.Exists(Path.Combine(wwwrootPath, DefaultCsvFileName)))
            result.Add("Current Season (epl.csv)");

        var files = Directory.GetFiles(wwwrootPath, "epl-*.csv");
        foreach (var f in files.OrderByDescending(f => f))
        {
            result.Add(Path.GetFileName(f));
        }
        return result;
    }

    private async Task<PredictionSession> PredictInternalAsync(
        List<MatchRaw> raw,
        int backtestRound = 0,
        int endBacktestRound = 0,
        string? wwwrootPath = null)
    {
        if (raw.Count == 0)
            return new PredictionSession { InfoMessage = "No data loaded from CSV." };

        raw = raw.OrderBy(m => m.RoundNumber).ThenBy(m => m.Date).ThenBy(m => m.MatchNumber).ToList();
        var hist = raw.Where(m => !string.IsNullOrWhiteSpace(m.Result)).ToList();
        var unplayed = raw.Where(m => string.IsNullOrWhiteSpace(m.Result)).ToList();

        if (hist.Count == 0 && unplayed.Count == 0)
            return new PredictionSession { InfoMessage = "No completed matches with scores found in CSV." };

        var lastCompletedRound = hist.Count > 0 ? hist.Max(m => m.RoundNumber) : 0;
        var nextRound = unplayed.Count > 0 ? unplayed.Min(m => m.RoundNumber) : lastCompletedRound;
        var fixturesNext = raw.Where(m => m.RoundNumber == nextRound && string.IsNullOrWhiteSpace(m.Result)).ToList();

        var tempDirObj = Directory.CreateTempSubdirectory("epl_predict_");
        var tempDir = tempDirObj.FullName;
        var csvPath = Path.Combine(tempDir, "epl.csv");

        WriteCsvToFile(raw, csvPath);

        // Copy all available historical season CSVs from wwwroot to temp directory
        var rootDir = wwwrootPath ?? Path.Combine(Directory.GetCurrentDirectory(), "wwwroot");
        var loadedSeasons = new List<string> { "Current Season" };
        int totalMatchesCount = hist.Count;

        if (Directory.Exists(rootDir))
        {
            var seasonFiles = Directory.GetFiles(rootDir, "epl-*.csv");
            foreach (var sFile in seasonFiles)
            {
                var fileName = Path.GetFileName(sFile);
                var destPath = Path.Combine(tempDir, fileName);
                File.Copy(sFile, destPath, overwrite: true);
                loadedSeasons.Add(fileName);
                try
                {
                    var lines = File.ReadAllLines(sFile).Length - 1;
                    totalMatchesCount += Math.Max(0, lines);
                }
                catch { }
            }
        }

        var scriptPath = Path.Combine(Directory.GetCurrentDirectory(), "predict.py");
        if (!File.Exists(scriptPath))
        {
            // Try parent or app domain
            scriptPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "predict.py");
        }

        // Determine Python command
        string pythonCmd = "python3";
        var venvPython = Path.Combine(Directory.GetCurrentDirectory(), ".venv", "bin", "python");
        if (File.Exists(venvPython))
            pythonCmd = venvPython;
        else if (OperatingSystem.IsWindows())
        {
            var venvWin = Path.Combine(Directory.GetCurrentDirectory(), ".venv", "Scripts", "python.exe");
            if (File.Exists(venvWin))
                pythonCmd = venvWin;
            else
                pythonCmd = "python";
        }

        string args;
        if (backtestRound > 0 && endBacktestRound > backtestRound)
            args = $"--backtest {backtestRound} {endBacktestRound}";
        else if (backtestRound > 0)
            args = $"--backtest {backtestRound} {backtestRound}";
        else
            args = "";

        var start = new ProcessStartInfo
        {
            FileName = pythonCmd,
            Arguments = string.IsNullOrEmpty(args) ? $"\"{scriptPath}\"" : $"\"{scriptPath}\" {args}",
            WorkingDirectory = tempDir,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        Process? process = null;
        try
        {
            process = Process.Start(start);
            if (process == null) throw new Exception($"Failed to start {pythonCmd}");
        }
        catch (Exception ex)
        {
            _log.LogWarning(ex, "Failed starting {PythonCmd}, attempting fallback", pythonCmd);
            start.FileName = OperatingSystem.IsWindows() ? "python" : "python3";
            try
            {
                process = Process.Start(start);
                if (process == null) throw new Exception("Failed starting fallback python");
            }
            catch (Exception ex2)
            {
                _log.LogError(ex2, "Fallback python execution failed");
                return new PredictionSession { InfoMessage = "Python runtime error: Ensure Python is installed." };
            }
        }

        await process.WaitForExitAsync();
        var output = await process.StandardOutput.ReadToEndAsync();
        var error = await process.StandardError.ReadToEndAsync();

        if (process.ExitCode != 0)
        {
            _log.LogError("Python script failed (exit {ExitCode}): {Error}", process.ExitCode, error);
            return new PredictionSession { InfoMessage = $"Prediction engine failed: {error}" };
        }

        var predsList = new List<FixturePrediction>();
        var predsCsvPath = Path.Combine(tempDir, "next_round_predictions.csv");
        if (File.Exists(predsCsvPath))
        {
            predsList = ParsePredictionsCsv(predsCsvPath, nextRound);
        }

        BacktestResult? backtest = null;
        if (!string.IsNullOrEmpty(output))
        {
            var backtestLine = output.Split('\n').FirstOrDefault(l => l.StartsWith("BACKTEST "));
            if (!string.IsNullOrEmpty(backtestLine))
            {
                try
                {
                    var json = backtestLine.Substring("BACKTEST ".Length);
                    var bt = JsonSerializer.Deserialize<JsonElement>(json);
                    backtest = new BacktestResult(
                        bt.GetProperty("round").GetInt32(),
                        bt.GetProperty("correct").GetInt32(),
                        bt.GetProperty("total").GetInt32(),
                        bt.GetProperty("accuracy").GetDouble(),
                        bt.GetProperty("items").EnumerateArray().Select(i => new BacktestItem(
                            i.GetProperty("HomeTeam").GetString() ?? "",
                            i.GetProperty("AwayTeam").GetString() ?? "",
                            DateTime.TryParse(i.GetProperty("Kickoff").GetString(), out var dt) ? dt : DateTime.MinValue,
                            i.GetProperty("Result").GetString() ?? "",
                            i.GetProperty("ActualOutcome").GetString() ?? "",
                            i.GetProperty("PredictedOutcome").GetString() ?? "",
                            (float)i.GetProperty("Probability").GetDouble(),
                            i.TryGetProperty("HomeWinProb", out var ph) ? (float)ph.GetDouble() : 0f,
                            i.TryGetProperty("DrawProb", out var pd) ? (float)pd.GetDouble() : 0f,
                            i.TryGetProperty("AwayWinProb", out var pa) ? (float)pa.GetDouble() : 0f
                        )).ToList()
                    );
                }
                catch (Exception ex)
                {
                    _log.LogError(ex, "Failed to parse backtest JSON from python stdout");
                }
            }
        }

        try { Directory.Delete(tempDir, recursive: true); } catch { }

        return new PredictionSession
        {
            LastCompletedRound = lastCompletedRound,
            NextRound = nextRound,
            NextRoundPredictions = predsList,
            Backtest = backtest,
            LoadedSeasons = loadedSeasons,
            TotalMatchesTrained = totalMatchesCount,
            InfoMessage = fixturesNext.Count == 0
                ? $"No upcoming fixtures for Round {nextRound}."
                : $"Analyzed {totalMatchesCount} matches across {loadedSeasons.Count} seasons. Predicting Round {nextRound}."
        };
    }

    private void WriteCsvToFile(List<MatchRaw> matches, string path)
    {
        using var writer = new StreamWriter(path);
        writer.WriteLine("Match Number,Round Number,Date,Location,Home Team,Away Team,Result");
        foreach (var m in matches)
        {
            writer.WriteLine($"{m.MatchNumber},{m.RoundNumber},{m.Date:dd/MM/yyyy HH:mm},{m.Location},{m.HomeTeam},{m.AwayTeam},{m.Result ?? ""}");
        }
    }

    private List<FixturePrediction> ParsePredictionsCsv(string path, int defaultRound)
    {
        var result = new List<FixturePrediction>();
        try
        {
            var lines = File.ReadAllLines(path);
            if (lines.Length <= 1) return result;

            var header = lines[0].Split(',');
            var headers = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < header.Length; i++) headers[header[i].Trim()] = i;

            for (int i = 1; i < lines.Length; i++)
            {
                var parts = lines[i].Split(',');
                if (parts.Length < header.Length) continue;

                var home = parts[headers["HomeTeam"]];
                var away = parts[headers["AwayTeam"]];
                var kickoff = DateTime.TryParse(parts[headers["Kickoff"]], out var dt) ? dt : DateTime.MinValue;
                var pred = parts[headers["PredictedOutcome"]];

                var pH = headers.ContainsKey("HomeWinProb") ? float.Parse(parts[headers["HomeWinProb"]], CultureInfo.InvariantCulture) : 0f;
                var pD = headers.ContainsKey("DrawProb") ? float.Parse(parts[headers["DrawProb"]], CultureInfo.InvariantCulture) : 0f;
                var pA = headers.ContainsKey("AwayWinProb") ? float.Parse(parts[headers["AwayWinProb"]], CultureInfo.InvariantCulture) : 0f;
                var xgH = headers.ContainsKey("ExpectedHomeGoals") ? float.Parse(parts[headers["ExpectedHomeGoals"]], CultureInfo.InvariantCulture) : 0f;
                var xgA = headers.ContainsKey("ExpectedAwayGoals") ? float.Parse(parts[headers["ExpectedAwayGoals"]], CultureInfo.InvariantCulture) : 0f;

                var maxProb = Math.Max(pH, Math.Max(pD, pA));
                if (maxProb <= 0f) maxProb = 0.5f;

                var round = defaultRound;
                if (headers.ContainsKey("Round")
                    && int.TryParse(parts[headers["Round"]], NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsedRound))
                {
                    round = parsedRound;
                }

                result.Add(new FixturePrediction(home, away, kickoff, round, pred, maxProb, pH, pD, pA, xgH, xgA));
            }
        }
        catch (Exception ex)
        {
            _log.LogError(ex, "Error parsing predictions CSV from {Path}", path);
        }
        return result;
    }

    private List<MatchRaw> LoadMatches(string path)
    {
        if (!File.Exists(path)) return new List<MatchRaw>();
        var lines = File.ReadAllLines(path);
        return ParseCsvLines(lines);
    }

    private List<MatchRaw> LoadMatchesFromText(string content)
    {
        var lines = content.Split(["\r\n", "\n"], StringSplitOptions.None);
        return ParseCsvLines(lines);
    }

    private List<MatchRaw> ParseCsvLines(IReadOnlyList<string> lines)
    {
        var result = new List<MatchRaw>();
        if (lines.Count <= 1) return result;

        var header = lines[0].Split(',');
        int idxMatch = Array.IndexOf(header, "Match Number");
        int idxRound = Array.IndexOf(header, "Round Number");
        int idxDate = Array.IndexOf(header, "Date");
        int idxLoc = Array.IndexOf(header, "Location");
        int idxHome = Array.IndexOf(header, "Home Team");
        int idxAway = Array.IndexOf(header, "Away Team");
        int idxRes = Array.IndexOf(header, "Result");

        for (int i = 1; i < lines.Count; i++)
        {
            var line = lines[i];
            if (string.IsNullOrWhiteSpace(line)) continue;
            var parts = line.Split(',');
            try
            {
                var m = new MatchRaw
                {
                    MatchNumber = idxMatch >= 0 && idxMatch < parts.Length ? int.Parse(parts[idxMatch]) : 0,
                    RoundNumber = idxRound >= 0 && idxRound < parts.Length ? int.Parse(parts[idxRound]) : 0,
                    Date = idxDate >= 0 && idxDate < parts.Length
                        ? DateTime.ParseExact(parts[idxDate].Trim(), "dd/MM/yyyy HH:mm", CultureInfo.InvariantCulture)
                        : DateTime.MinValue,
                    Location = idxLoc >= 0 && idxLoc < parts.Length ? parts[idxLoc].Trim() : "",
                    HomeTeam = idxHome >= 0 && idxHome < parts.Length ? parts[idxHome].Trim() : "",
                    AwayTeam = idxAway >= 0 && idxAway < parts.Length ? parts[idxAway].Trim() : "",
                    Result = (idxRes >= 0 && idxRes < parts.Length && !string.IsNullOrWhiteSpace(parts[idxRes])) ? parts[idxRes].Trim() : null
                };
                result.Add(m);
            }
            catch { }
        }
        return result;
    }

    private void ArchiveCurrentSeasonIfRolled(string wwwrootPath, int incomingSeasonYear)
    {
        var localYear = GetLocalCsvSeasonYear(wwwrootPath);
        if (localYear is null || localYear.Value == incomingSeasonYear)
            return;

        var filePath = Path.Combine(wwwrootPath, DefaultCsvFileName);
        var archivePath = Path.Combine(wwwrootPath, $"epl-{localYear.Value}.csv");
        File.Copy(filePath, archivePath, overwrite: true);
        _log.LogInformation("Archived finished season {Year} to {Path}", localYear.Value, archivePath);
    }

    private static int? GetSeasonYearFromMatches(List<MatchRaw> matches)
    {
        var first = matches.Where(m => m.Date != DateTime.MinValue).OrderBy(m => m.Date).FirstOrDefault();
        return first == null ? null : DeriveSeasonYear(first.Date);
    }

    private static HttpClient CreateHttpClient(int timeoutSeconds)
    {
        return new HttpClient { Timeout = TimeSpan.FromSeconds(timeoutSeconds) };
    }

    private static string FixtureDownloadUrl(int year) =>
        $"https://fixturedownload.com/download/epl-{year}-GMTStandardTime.csv";

    private static bool LooksLikeFixtureCsv(string? content) =>
        !string.IsNullOrWhiteSpace(content) && content.Contains("Match Number");

    private async Task<bool> SeasonCsvExistsAsync(HttpClient client, int year, CancellationToken ct)
    {
        try
        {
            using var resp = await client.GetAsync(FixtureDownloadUrl(year), ct);
            if (!resp.IsSuccessStatusCode)
                return false;
            var body = await resp.Content.ReadAsStringAsync(ct);
            return LooksLikeFixtureCsv(body);
        }
        catch
        {
            return false;
        }
    }
}
