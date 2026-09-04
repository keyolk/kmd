# kmd — build & evaluation
#
# 검색 품질/효용성 평가 타깃. 자세한 배경은 eval/README.md 참고.

BIN := target/release/kmd
CARGO := cargo
PREFIX := $(HOME)/.local
INSTALL_BIN := $(PREFIX)/bin/kmd

# 벡터 검색(embeddinggemma)은 llama-cpp-2를 링크하므로 빌드가 ~1.5분 걸린다.
# `make build`는 가볍게 두고, 벡터가 필요한 타깃만 FEATURES를 붙인다.
FEATURES := --features embed

.PHONY: build
build:
	$(CARGO) build --release

## 벡터 검색 포함 빌드 (kmd embed / vsearch / query)
.PHONY: build-embed
build-embed:
	$(CARGO) build --release $(FEATURES)

# Copy the binary into $(PREFIX)/bin. Real file copy, not a symlink.
# 벡터 검색을 포함해 설치한다. RAG 훅이 hybrid를 쓰려면 데몬 바이너리에
# embed 피처가 있어야 하고, 없으면 BM25로 조용히 내려가 개선이 사라진다.
.PHONY: install
install: build-embed
	@mkdir -p $(PREFIX)/bin
	@rm -f $(INSTALL_BIN)
	@cp $(BIN) $(INSTALL_BIN)
	@chmod +x $(INSTALL_BIN)
	@echo "installed -> $(INSTALL_BIN)"
	@echo "restart the daemon so it picks up the new binary:"
	@echo "  launchctl kickstart -k gui/$$(id -u)/ai.kmd.daemon"

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

# ── 엔진 대조: BM25 vs 벡터 vs hybrid ───────────────────────────────
#
# 세 엔진을 같은 쿼리·같은 컬렉션 범위로 돌린다. 범위를 좁히는 이유는
# eval.rs의 주석 참고 — project 축은 임베딩 대상이 아니라서, 범위를 그대로
# 두면 벡터만 경쟁 문서가 없는 모집단에서 검색한다.

## known-item 3엔진 대조
.PHONY: eval-compare
eval-compare: build-embed
	$(BIN) eval --sample 200 -k 5 --compare

## gold set 3엔진 대조
.PHONY: eval-compare-gold
eval-compare-gold: build-embed
	$(BIN) eval --gold eval/gold.yaml -k 5 --compare

# ── L2: 주입 채택률 (utilization) ────────────────────────────────────

## rag.jsonl + transcript 조인 — 답변이 주입 컨텍스트를 실제로 썼나
.PHONY: util
util: build
	$(BIN) util

## 전체 평가 한 번에
.PHONY: eval-all
eval-all: eval util
