# Enterprise Support Agent (FastAPI + LangGraph + PostgreSQL)

```
.
├── agent/
│   ├── graph.py      # StateGraph, routing, Postgres checkpointer lifecycle
│   ├── state.py      # AgentState (add_messages reducer)
│   └── tools.py      # crm_lookup, calculator, analyze_data, create_support_ticket
├── api/
│   ├── main.py       # FastAPI app, DI, error handling
│   └── schemas.py    # Pydantic v2 request/response models
├── core/config.py    # pydantic-settings configuration
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Run

```bash
cp .env.example .env          # set OPENAI_API_KEY and POSTGRES_PASSWORD
docker compose up --build
```

```bash
curl -s localhost:8000/api/v1/chat -H 'content-type: application/json' -d '{
  "thread_id": "demo-1",
  "message": "Look up Globex Retail and open a high priority ticket about slow dashboards."
}'

# Continue the same conversation (state restored from Postgres):
curl -s localhost:8000/api/v1/chat -H 'content-type: application/json' -d '{
  "thread_id": "demo-1", "message": "What plan are they on?"
}'

curl -s localhost:8000/api/v1/threads/demo-1/history
```

Interactive docs: http://localhost:8000/docs

## Production notes

- Per-turn LLM budget (`AGENT_MAX_ITERATIONS`) and `AGENT_RECURSION_LIMIT` stop runaway loops.
- Serialise concurrent requests per `thread_id` (client, gateway, or advisory lock).
- Add authentication (API key / JWT dependency) before exposing publicly.
- Replace the in-memory CRM and ticket stores in `agent/tools.py` with real clients.
- Pin exact dependency versions via a lockfile for reproducible builds.