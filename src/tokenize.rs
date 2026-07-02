//! 토크나이저 구성 — 한글은 lindera ko-dic 형태소 분석, 영어는 lowercase+stemmer.
//! 같은 텍스트를 *_ko / *_en 두 필드에 넣고 두 분석기로 각각 인덱싱한다.

use anyhow::Result;
use lindera::dictionary::{DictionaryKind, load_embedded_dictionary};
use lindera::mode::Mode;
use lindera::segmenter::Segmenter;
use lindera_tantivy::tokenizer::LinderaTokenizer;
use tantivy::Index;
use tantivy::tokenizer::{Language, LowerCaser, SimpleTokenizer, Stemmer, TextAnalyzer};

pub const KO_TOKENIZER: &str = "ko_lindera";
pub const EN_TOKENIZER: &str = "en_stem_lc";

pub fn register(index: &Index) -> Result<()> {
    // 한글: ko-dic 형태소 분석 ("인증서를" → "인증서" + "를")
    let dictionary = load_embedded_dictionary(DictionaryKind::KoDic)?;
    let segmenter = Segmenter::new(Mode::Normal, dictionary, None);
    let ko = LinderaTokenizer::from_segmenter(segmenter);
    index.tokenizers().register(KO_TOKENIZER, ko);

    // 영어: 단순 분리 + lowercase + porter stemmer
    let en = TextAnalyzer::builder(SimpleTokenizer::default())
        .filter(LowerCaser)
        .filter(Stemmer::new(Language::English))
        .build();
    index.tokenizers().register(EN_TOKENIZER, en);

    Ok(())
}
