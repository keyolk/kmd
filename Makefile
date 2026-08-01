# kmd — build & evaluation
#
# 검색 품질/효용성 평가 타깃. 자세한 배경은 eval/README.md 참고.

BIN := target/release/kmd
CARGO := cargo

.PHONY: build
build:
	$(CARGO) build --release

# ── L1: 검색 품질 (retrieval quality) ────────────────────────────────

## known-item 자기지도 평가 — kmd vs qmd, 한/영 분리. 라벨 불필요.
.PHONY: eval
eval: build
	$(BIN) eval --sample 400 -k 5 --compare-qmd

## 한글만 격리 측정 (kmd의 핵심 가치)
.PHONY: eval-ko
eval-ko: build
	$(BIN) eval --sample 400 -k 5 --hangul --compare-qmd

## 수동 라벨 gold set 회귀 (전 인덱스 retrievability)
.PHONY: eval-gold
eval-gold: build
	$(BIN) eval --gold eval/gold.yaml -k 5 --compare-qmd

# ── L2: 주입 채택률 (utilization) ────────────────────────────────────

## rag.jsonl + transcript 조인 — 답변이 주입 컨텍스트를 실제로 썼나
.PHONY: util
util: build
	$(BIN) util

# ── L3: A/B 블라인드 비교 ────────────────────────────────────────────

## 내장 프록시 판정 (프롬프트 키워드 커버리지)
.PHONY: ab
ab: build
	$(BIN) ab --prompts eval/prompts.txt

## 블라인드 페어 생성 → 외부 LLM/사람 판정용
.PHONY: ab-emit
ab-emit: build
	$(BIN) ab --prompts eval/prompts.txt --emit eval/pairs.jsonl

## 전체 평가 한 번에
.PHONY: eval-all
eval-all: eval util ab
