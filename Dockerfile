FROM golang:1.21-alpine AS builder
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /bin/anchor-cli ./cmd/anchor-cli

FROM alpine:3.19
RUN apk add --no-cache ca-certificates curl
COPY --from=builder /bin/anchor-cli /usr/local/bin/anchor-cli
EXPOSE 8080 9090 9091
ENTRYPOINT ["anchor-cli"]
CMD ["server", "--config", "/etc/anchor/config.yaml"]
