# Build stage
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /app

# 1) Install Python + venv tooling (+ libgomp for sklearn/xgboost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# 2) Create venv and install python deps into it (NOT system-wide)
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only requirements first (better Docker cache)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 3) Copy the rest and build .NET
COPY . .
RUN dotnet publish -c Release -o /out

# Runtime stage
FROM mcr.microsoft.com/dotnet/aspnet:10.0 AS runtime
WORKDIR /app

# If your app runs python at runtime, you need python runtime + libgomp here too
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Bring the venv forward
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Bring the published app forward
COPY --from=build /out ./

COPY --from=build /app/predict.py /app/predict.py

ENV ASPNETCORE_URLS=http://0.0.0.0:$PORT
EXPOSE 80

ENTRYPOINT ["dotnet", "PredictorBlazor.dll"]
