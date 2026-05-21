.PHONY: all build test lint generate clean docker run-standalone demo

BINARY := anchor-cli
CMD     := ./cmd/anchor-cli

all: build

build:
	go build -ldflags="-s -w" -o bin/$(BINARY) $(CMD)

test:
	go test ./... -race -timeout 60s

lint:
	golangci-lint run ./...

generate:
	@which buf > /dev/null 2>&1 && buf generate || \
	 protoc --go_out=. --go_opt=paths=source_relative \
	        --go-grpc_out=. --go-grpc_opt=paths=source_relative \
	        proto/anchor/v1/anchor.proto

clean:
	rm -rf bin/ gen/

docker:
	docker build -t bastion/anchor:dev .

run-standalone:
	go run $(CMD) server --port 8080

demo:
	go run $(CMD) demo

interactive:
	go run $(CMD) interactive

tidy:
	go mod tidy

.DEFAULT_GOAL := build
