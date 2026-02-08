using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace PredictorBlazor.Services;

public record FixturePrediction(
    string HomeTeam,
    string AwayTeam,
    DateTime Kickoff,
    int Round,
    string PredictedOutcome,
    float Probability
);

public record BacktestItem(
    string HomeTeam,
    string AwayTeam,
    DateTime Kickoff,
    string Result,
    string ActualOutcome,
    string PredictedOutcome,
    float Probability
);

public record BacktestResult(int Round, int Correct, int Total, double Accuracy, List<BacktestItem> Items);

public class PredictionSession
{
    public int LastCompletedRound { get; init; }
    public int NextRound { get; init; }
    public List<FixturePrediction> NextRoundPredictions { get; init; } = new();
    public BacktestResult? Backtest { get; init; }
    public string InfoMessage { get; init; } = string.Empty;
}

public class PredictorService
{
    private readonly ILogger<PredictorService> _log;

    public PredictorService(ILogger<PredictorService> log)
    {
        _log = log;
    }
    // Raw match row as loaded from CSV
    public class MatchRaw
    {
        public int MatchNumber { get; set; }
        public int RoundNumber { get; set; }
        public DateTime Date { get; set; }
        public string HomeTeam { get; set; } = "";
        public string AwayTeam { get; set; } = "";
        public string Location { get; set; } = "";
        public string? Result { get; set; } // can be null/empty
    }

    // Internal stats per team
    public class TeamStats
    {
        public List<int> Gf { get; } = new();
        public List<int> Ga { get; } = new();
        public List<int> Pts { get; } = new();
        public List<int> W { get; } = new();
        public List<int> D { get; } = new();
        public List<int> L { get; } = new();
    }

    // Elo rating helper
    public class EloTable
    {
        private readonly double _base;
        private readonly double _k;
        private readonly double _homeAdv;
        private readonly Dictionary<string, double> _ratings = new();

        public EloTable(double @base = 1500.0, double k = 24.0, double homeAdv = 65.0)
        {
            _base = @base;
            _k = k;
            _homeAdv = homeAdv;
        }

        public double Get(string team) => _ratings.GetValueOrDefault(team, _base);

        private double Expect(double ra, double rb, bool isHome)
        {
            var adj = isHome ? _homeAdv : -_homeAdv;
            return 1.0 / (1.0 + Math.Pow(10.0, ((rb - (ra + adj)) / 400.0)));
        }

        public void Update(string home, string away, int hg, int ag)
        {
            var ra = Get(home);
            var rb = Get(away);
            var ea = Expect(ra, rb, true);
            var eb = 1.0 - ea;

            double sa, sb;
            if (hg > ag) { sa = 1.0; sb = 0.0; }
            else if (hg < ag) { sa = 0.0; sb = 1.0; }
            else { sa = 0.5; sb = 0.5; }

            _ratings[home] = ra + _k * (sa - ea);
            _ratings[away] = rb + _k * (sb - eb);
        }
    }

    // Training / prediction feature row
    public class MatchFeature
    {
        // label
        public string Outcome { get; set; } = ""; // "H","D","A"

        // features
        public float EloHome { get; set; }
        public float EloAway { get; set; }
        public float EloDiff { get; set; }

        public float HomeGf { get; set; }
        public float HomeGa { get; set; }
        public float HomePts { get; set; }
        public float HomeW { get; set; }
        public float HomeD { get; set; }
        public float HomeL { get; set; }

        public float AwayGf { get; set; }
        public float AwayGa { get; set; }
        public float AwayPts { get; set; }
        public float AwayW { get; set; }
        public float AwayD { get; set; }
        public float AwayL { get; set; }

        public float GfDiff { get; set; }
        public float GaDiff { get; set; }
        public float PtsDiff { get; set; }

        public float HomeId { get; set; }
        public float AwayId { get; set; }
        public float TeamIdDiff { get; set; }
        public float LocId { get; set; }

        public float Round { get; set; }
        public float Dow { get; set; }
        public float Month { get; set; }

        // For reference / later joins
        public string HomeTeam { get; set; } = "";
        public string AwayTeam { get; set; } = "";
        public DateTime Kickoff { get; set; }
        public int RoundNumber { get; set; }
        public string Location { get; set; } = "";
    }



    private const string DefaultCsvFileName = "epl.csv";

    public async Task<PredictionSession> PredictAsync(Stream csvStream, int backtestRound = 9, CancellationToken ct = default)
    {
        var sw = Stopwatch.StartNew();
        _log.LogInformation("PredictAsync: starting. backtestRound={BacktestRound}", backtestRound);
        using var reader = new StreamReader(csvStream, leaveOpen: true);
        var csvText = await reader.ReadToEndAsync(ct);
        _log.LogDebug("PredictAsync: read {Length} chars from stream", csvText.Length);
        var raw = LoadMatchesFromText(csvText);
        var result = await PredictInternalAsync(raw, backtestRound);
        sw.Stop();
        _log.LogInformation("PredictAsync: finished in {ElapsedMs} ms. LastCompletedRound={Last}, NextRound={Next}, Fixtures={Fixtures}",
            sw.ElapsedMilliseconds, result.LastCompletedRound, result.NextRound, result.NextRoundPredictions?.Count ?? 0);
        return result;
    }

    public async Task<PredictionSession> PredictFromFileIfExistsAsync(string wwwrootPath, int backtestRound = 9)
    {
        var sw = Stopwatch.StartNew();
        var filePath = Path.Combine(wwwrootPath, DefaultCsvFileName);
        _log.LogInformation("PredictFromFileIfExistsAsync: looking for {Path}", filePath);
        if (!File.Exists(filePath))
        {
            _log.LogWarning("PredictFromFileIfExistsAsync: file not found at {Path}", filePath);
            return new PredictionSession { InfoMessage = $"No {DefaultCsvFileName} found under wwwroot." };
        }
        var raw = LoadMatches(filePath);
        var result = await PredictInternalAsync(raw, backtestRound);
        sw.Stop();
        _log.LogInformation("PredictFromFileIfExistsAsync: finished in {ElapsedMs} ms.", sw.ElapsedMilliseconds);
        return result;
    }

    public async Task UpdateFixturesAsync(string wwwrootPath)
    {
        var url = "https://fixturedownload.com/download/epl-2025-GMTStandardTime.csv";
        var filePath = Path.Combine(wwwrootPath, DefaultCsvFileName);
        _log.LogInformation("UpdateFixturesAsync: downloading from {Url} to {Path}", url, filePath);

        using var client = new HttpClient();
        var csvContent = await client.GetStringAsync(url);
        
        if (string.IsNullOrWhiteSpace(csvContent))
        {
            throw new Exception("Downloaded CSV content is empty.");
        }

        // Basic validation: check header
        if (!csvContent.StartsWith("Match Number,Round Number,Date,Location,Home Team,Away Team,Result"))
        {
             _log.LogWarning("UpdateFixturesAsync: Unexpected header in downloaded CSV. Proceeding anyway but logging warning.");
        }

        await File.WriteAllTextAsync(filePath, csvContent);
        _log.LogInformation("UpdateFixturesAsync: successfully updated {Path}", filePath);
    }

    private async Task<PredictionSession> PredictInternalAsync(List<MatchRaw> raw, int backtestRound)
    {
        _log.LogDebug("PredictInternalAsync: raw rows={Count}", raw.Count);
        if (raw.Count == 0)
        {
            _log.LogWarning("PredictInternalAsync: no rows in dataset");
            return new PredictionSession { InfoMessage = "No data loaded from epl.csv" };
        }

        // Sort
        raw = raw.OrderBy(m => m.RoundNumber).ThenBy(m => m.Date).ThenBy(m => m.MatchNumber).ToList();

        var hist = raw.Where(m => !string.IsNullOrWhiteSpace(m.Result)).ToList();
        _log.LogDebug("PredictInternalAsync: history rows with results={Count}", hist.Count);
        if (hist.Count == 0)
        {
            _log.LogWarning("PredictInternalAsync: no completed matches in dataset");
            return new PredictionSession { InfoMessage = "No completed matches with results in CSV" };
        }

        var lastCompletedRound = hist.Max(m => m.RoundNumber);
        var nextRound = lastCompletedRound + 1;
        var fixturesNext = raw.Where(m => m.RoundNumber == nextRound && string.IsNullOrWhiteSpace(m.Result)).ToList();
        _log.LogInformation("PredictInternalAsync: lastCompletedRound={Last} nextRound={Next} fixturesNext={Fixtures}", lastCompletedRound, nextRound, fixturesNext.Count);

        // Save CSV to temp dir
        var tempDir = Path.GetTempPath();
        var csvPath = Path.Combine(tempDir, "epl.csv");
        
        // Write the raw matches to the CSV in the temp directory so the Python script can read it
        WriteCsvToFile(raw, csvPath);
        
        // Copy 2024 season data to temp dir if it exists for multi-season training
        var csv2024Source = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", "epl-2024.csv");
        var csv2024Dest = Path.Combine(tempDir, "epl-2024.csv");
        if (File.Exists(csv2024Source))
        {
            _log.LogInformation("Copying 2024 season data to temp directory for enhanced training");
            File.Copy(csv2024Source, csv2024Dest, overwrite: true);
        }
        else
        {
            _log.LogWarning("2024 season data not found at {Path}. Model will train on current season only.", csv2024Source);
        }

        var args = backtestRound > 0 ? backtestRound.ToString() : "";
        var scriptPath = Path.Combine(Directory.GetCurrentDirectory(), "predict.py");
        _log.LogDebug("scriptPath: {ScriptPath}", scriptPath);
        _log.LogDebug("AppDomain.BaseDirectory: {Base}", AppDomain.CurrentDomain.BaseDirectory);
        _log.LogDebug("CurrentDirectory: {Curr}", Directory.GetCurrentDirectory());

        // Determine Python executable
        string pythonCmd;
        var venvPath = Path.Combine(Directory.GetCurrentDirectory(), ".venv");
        if (OperatingSystem.IsWindows())
        {
            var venvPython = Path.Combine(venvPath, "Scripts", "python.exe");
            pythonCmd = File.Exists(venvPython) ? venvPython : "python";
        }
        else
        {
            var venvPython = Path.Combine(venvPath, "bin", "python");
            pythonCmd = File.Exists(venvPython) ? venvPython : "python3";
        }
        
        _log.LogInformation("Using Python command: {Cmd}", pythonCmd);

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
            if (process == null)
            {
                throw new Exception($"Failed to start {pythonCmd}");
            }
        }
        catch (Exception ex)
        {
            _log.LogWarning(ex, "{PythonCmd} failed to start, trying fallback", pythonCmd);
            // Fallback logic if venv failed or wasn't found and default failed
            if (pythonCmd != "python" && pythonCmd != "python3")
            {
                 pythonCmd = OperatingSystem.IsWindows() ? "python" : "python3";
                 start.FileName = pythonCmd;
                 try 
                 {
                    process = Process.Start(start);
                    if (process == null) throw new Exception($"Failed to start fallback {pythonCmd}");
                 }
                 catch (Exception ex2)
                 {
                     _log.LogError(ex2, "Fallback python failed");
                     return new PredictionSession { InfoMessage = "Python not found. Please run setup script." };
                 }
            }
            else
            {
                 return new PredictionSession { InfoMessage = $"Python execution failed: {ex.Message}" };
            }
        }

        await process.WaitForExitAsync();
        var output = await process.StandardOutput.ReadToEndAsync();
        var error = await process.StandardError.ReadToEndAsync();

        _log.LogInformation("Python process exited with code {ExitCode}", process.ExitCode);
        if (!string.IsNullOrWhiteSpace(output)) _log.LogDebug("Python stdout: {Output}", output);
        if (!string.IsNullOrWhiteSpace(error)) _log.LogDebug("Python stderr: {Error}", error);

        if (process.ExitCode != 0)
        {
            _log.LogError("Python script failed with exit code {ExitCode}: {Error}", process.ExitCode, error);
            return new PredictionSession { InfoMessage = $"Python execution failed: {error}. Ensure Python and dependencies are installed (run setup.sh)." };
        }

        // Parse predictions CSV
        var predsList = new List<FixturePrediction>();
        var predsCsvPath = Path.Combine(tempDir, "next_round_predictions.csv");
        if (File.Exists(predsCsvPath))
        {
            predsList = ParsePredictionsCsv(predsCsvPath);
        }

        // Parse backtest from stdout if present
        BacktestResult? backtest = null;
        if (!string.IsNullOrEmpty(output))
        {
            var backtestLine = output.Split('\n').FirstOrDefault(l => l.StartsWith("BACKTEST"));
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
                            i.GetProperty("HomeTeam").GetString()!,
                            i.GetProperty("AwayTeam").GetString()!,
                            i.GetProperty("Kickoff").GetString()!.Length > 0 ? DateTime.Parse(i.GetProperty("Kickoff").GetString()!) : DateTime.MinValue,
                            i.GetProperty("Result").GetString()!,
                            i.GetProperty("ActualOutcome").GetString()!,
                            i.GetProperty("PredictedOutcome").GetString()!,
                            i.GetProperty("Probability").GetSingle()
                        )).ToList()
                    );
                }
                catch (Exception ex)
                {
                     _log.LogError(ex, "Failed to parse backtest JSON");
                }
            }
        }

        return new PredictionSession
        {
            LastCompletedRound = lastCompletedRound,
            NextRound = nextRound,
            NextRoundPredictions = predsList,
            Backtest = backtest,
            InfoMessage = fixturesNext.Count == 0
                ? $"No fixtures found for round {nextRound}. Nothing to predict."
                : $"Last completed round: {lastCompletedRound} | Predicting round {nextRound}"
        };
    }

    private void WriteCsvToFile(List<MatchRaw> matches, string path)
    {
        using var writer = new StreamWriter(path);
        writer.WriteLine("Match Number,Round Number,Date,Home Team,Away Team,Location,Result");
        foreach (var m in matches)
        {
            writer.WriteLine($"{m.MatchNumber},{m.RoundNumber},{m.Date:dd/MM/yyyy HH:mm},{m.HomeTeam},{m.AwayTeam},{m.Location},{m.Result ?? ""}");
        }
    }

    private List<FixturePrediction> ParsePredictionsCsv(string path)
    {
        var result = new List<FixturePrediction>();
        var lines = File.ReadAllLines(path);
        if (lines.Length <= 1) return result;

        var header = lines[0].Split(',');
        var headers = new Dictionary<string, int>();
        for (int i = 0; i < header.Length; i++) headers[header[i]] = i;

        for (int i = 1; i < lines.Length; i++)
        {
            var parts = lines[i].Split(',');
            if (parts.Length < headers.Count) continue;

            var home = parts[headers["Home Team"]];
            var away = parts[headers["Away Team"]];
            var kickoff = DateTime.Parse(parts[headers["Date"]]);
            var round = int.Parse(parts[headers["Round Number"]]);
            var pred = parts[headers["PredictedOutcome"]];
            var maxProb = Math.Max(float.Parse(parts[headers["pH"]]), Math.Max(float.Parse(parts[headers["pD"]]), float.Parse(parts[headers["pA"]])));
            result.Add(new FixturePrediction(home, away, kickoff, round, pred, maxProb));
        }

        return result;
    }

    // ------------------------------
    // CSV loader
    // ------------------------------
    private List<MatchRaw> LoadMatches(string path)
    {
        var result = new List<MatchRaw>();
        if (!File.Exists(path)) return result;
        _log.LogInformation("LoadMatches: reading file {Path}", path);
        var lines = File.ReadAllLines(path);
        _log.LogDebug("LoadMatches: {Lines} lines read", lines.Length);
        return ParseCsvLines(lines);
    }

    private List<MatchRaw> LoadMatchesFromText(string content)
    {
        var lines = content.Split(["\r\n", "\n"], StringSplitOptions.None);
        _log.LogDebug("LoadMatchesFromText: split into {Lines} lines", lines.Length);
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
        int idxHome = Array.IndexOf(header, "Home Team");
        int idxAway = Array.IndexOf(header, "Away Team");
        int idxLoc = Array.IndexOf(header, "Location");
        int idxRes = Array.IndexOf(header, "Result");
        _log.LogDebug("ParseCsvLines: header parsed. Columns -> Match:{Match} Round:{Round} Date:{Date} Home:{Home} Away:{Away} Loc:{Loc} Res:{Res}",
            idxMatch, idxRound, idxDate, idxHome, idxAway, idxLoc, idxRes);

        for (int i = 1; i < lines.Count; i++)
        {
            var line = lines[i];
            if (string.IsNullOrWhiteSpace(line)) continue;
            var parts = line.Split(','); // assumes no commas inside fields
            try
            {
                var m = new MatchRaw
                {
                    MatchNumber = idxMatch >= 0 && idxMatch < parts.Length ? int.Parse(parts[idxMatch]) : 0,
                    RoundNumber = idxRound >= 0 && idxRound < parts.Length ? int.Parse(parts[idxRound]) : 0,
                    Date = idxDate >= 0 && idxDate < parts.Length
                        ? DateTime.ParseExact(parts[idxDate], "dd/MM/yyyy HH:mm", CultureInfo.InvariantCulture)
                        : DateTime.MinValue,
                    HomeTeam = idxHome >= 0 && idxHome < parts.Length ? parts[idxHome] : "",
                    AwayTeam = idxAway >= 0 && idxAway < parts.Length ? parts[idxAway] : "",
                    Location = idxLoc >= 0 && idxLoc < parts.Length ? parts[idxLoc] : "",
                    Result = (idxRes >= 0 && idxRes < parts.Length && !string.IsNullOrWhiteSpace(parts[idxRes])) ? parts[idxRes] : null
                };
                result.Add(m);
            }
            catch (Exception ex)
            {
                _log.LogWarning(ex, "ParseCsvLines: skipping bad line {LineNo}: {Line}", i + 1, line);
            }
        }
        _log.LogInformation("ParseCsvLines: parsed {Count} rows (excluding header)", result.Count);
        return result;
    }
}
