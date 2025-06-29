build:
	docker build -t collateral-agent .

run:
	docker run -p 8000:8000 -v "$(pwd)/datasets:/app/datasets" collateral-agent

test:
	curl -X POST http://localhost:8000/generate_report \
	  -H "Content-Type: application/json" \
	  -d '{"asset_id": "157515_v2"}'