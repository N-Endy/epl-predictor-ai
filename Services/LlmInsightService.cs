using System.Net.Http.Json;
using System.Text.Json;

namespace PredictorBlazor.Services;

public class LlmInsightService
{
    private readonly IConfiguration _config;
    private readonly ILogger<LlmInsightService> _log;
    private readonly IHttpClientFactory _clientFactory;

    public LlmInsightService(IConfiguration config, ILogger<LlmInsightService> log, IHttpClientFactory clientFactory)
    {
        _config = config;
        _log = log;
        _clientFactory = clientFactory;
    }

    public bool IsEnabled => _config.GetValue<bool>("Llm:Enabled") && 
                             !string.IsNullOrWhiteSpace(_config.GetValue<string>("Llm:ApiKey"));

    public async Task<string> GetTacticalInsightAsync(FixturePrediction p, CancellationToken ct = default)
    {
        if (!IsEnabled)
        {
            return p.GenerateTacticalSummary();
        }

        try
        {
            var apiKey = _config.GetValue<string>("Llm:ApiKey");
            var endpoint = _config.GetValue<string>("Llm:Endpoint");
            if (string.IsNullOrWhiteSpace(endpoint))
                endpoint = "https://api.openai.com/v1/chat/completions";

            var model = _config.GetValue<string>("Llm:Model") ?? "gpt-4o-mini";

            var outcome = p.PredictedOutcome switch
            {
                "H" => $"{p.HomeTeam} Win",
                "A" => $"{p.AwayTeam} Win",
                _ => "Draw"
            };

            var prompt = $"You are an elite EPL tactical analyst. In 2 sharp sentences, summarize why the statistical model predicts {outcome} ({p.Probability:P0} confidence) for {p.HomeTeam} vs {p.AwayTeam} (xG: {p.ExpectedHomeGoals:0.0} - {p.ExpectedAwayGoals:0.0}). Highlight rating superiority or fatigue.";

            var requestBody = new
            {
                model = model,
                messages = new[]
                {
                    new { role = "system", content = "You are a concise, sharp Premier League analytics writer." },
                    new { role = "user", content = prompt }
                },
                max_tokens = 90,
                temperature = 0.6
            };

            var client = _clientFactory.CreateClient();
            client.Timeout = TimeSpan.FromSeconds(5);

            using var request = new HttpRequestMessage(HttpMethod.Post, endpoint);
            request.Headers.Add("Authorization", $"Bearer {apiKey}");
            request.Content = JsonContent.Create(requestBody);

            using var resp = await client.SendAsync(request, ct);
            if (resp.IsSuccessStatusCode)
            {
                var doc = await resp.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: ct);
                var content = doc.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString();
                if (!string.IsNullOrWhiteSpace(content))
                    return content.Trim();
            }
        }
        catch (Exception ex)
        {
            _log.LogWarning(ex, "External LLM call failed or timed out; falling back to deterministic tactical summary");
        }

        return p.GenerateTacticalSummary();
    }
}
