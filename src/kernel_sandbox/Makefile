.PHONY: build test vet race

GOCACHE := $(CURDIR)/.cache/go-build
GOTMPDIR := $(CURDIR)/.cache/go-tmp

build:
	@mkdir -p bin $(GOCACHE) $(GOTMPDIR)
	GOCACHE=$(GOCACHE) GOTMPDIR=$(GOTMPDIR) CGO_ENABLED=0 go build -o bin/kernel-sandbox-master ./cmd/kernel-sandbox-master
	GOCACHE=$(GOCACHE) GOTMPDIR=$(GOTMPDIR) CGO_ENABLED=0 go build -o bin/gpu-pool-monitor ./cmd/gpu-pool-monitor
	GOCACHE=$(GOCACHE) GOTMPDIR=$(GOTMPDIR) CGO_ENABLED=0 go build -o bin/sandboxd ./cmd/sandboxd

test:
	@mkdir -p $(GOCACHE) $(GOTMPDIR)
	GOCACHE=$(GOCACHE) GOTMPDIR=$(GOTMPDIR) go test ./...

vet:
	@mkdir -p $(GOCACHE) $(GOTMPDIR)
	GOCACHE=$(GOCACHE) GOTMPDIR=$(GOTMPDIR) go vet ./...

race:
	@mkdir -p $(GOCACHE) $(GOTMPDIR)
	GOCACHE=$(GOCACHE) GOTMPDIR=$(GOTMPDIR) CGO_ENABLED=1 go test -race ./...
