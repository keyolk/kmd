# kmd — build & evaluation
#
# 검색 품질/효용성 평가 타깃. 자세한 배경은 eval/README.md 참고.

BIN := target/release/kmd
CARGO := cargo
PREFIX := $(HOME)/.local
INSTALL_BIN := $(PREFIX)/bin/kmd

.PHONY: build
build:
	$(CARGO) build --release

# Copy the binary into $(PREFIX)/bin. Real file copy, not a symlink.
.PHONY: install
install: build
	@mkdir -p $(PREFIX)/bin
	@rm -f $(INSTALL_BIN)
	@cp $(BIN) $(INSTALL_BIN)
	@chmod +x $(INSTALL_BIN)
	@echo "installed -> $(INSTALL_BIN)"

# ── L1: 검색 품질 (retrieval quality) ────────────────────────────────

## known-item 자기지도 평가 — 한/영 분리. 라벨 불필요.
.PHONY: eval
eval: build
	$(BIN) eval --sample 400 -k 5

## 한글만 격리 측정 (kmd의 핵심 가치)
.PHONY: eval-ko
eval-ko: build
	$(BIN) eval --sample 400 -k 5 --hangul

## 수동 라벨 gold set 회귀 (전 인덱스 retrievability)
.PHONY: eval-gold
eval-gold: build
	$(BIN) eval --gold eval/gold.yaml -k 5

# ── L2: 주입 채택률 (utilization) ────────────────────────────────────

## rag.jsonl + transcript 조인 — 답변이 주입 컨텍스트를 실제로 썼나
.PHONY: util
util: build
	$(BIN) util

## 전체 평가 한 번에
.PHONY: eval-all
eval-all: eval util
