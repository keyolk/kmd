//! SQLite 문서 스토어 — 스캔 상태(mtime/hash)와 문서 본문을 보관.
//! tantivy 인덱스는 파생물이며, 여기가 source of truth.

use anyhow::Result;
use rusqlite::{Connection, params};
use std::path::Path;

pub struct Store {
    pub conn: Connection,
}

#[derive(Debug, Clone)]
pub struct Doc {
    pub id: i64,
    pub collection: String,
    /// 컬렉션 루트 기준 상대 경로
    pub relpath: String,
    pub title: String,
    pub body: String,
    pub context: Option<String>,
    /// 본문 미저장 문서의 원본 절대경로. Some이면 `body`는 비어 있고
    /// 본문이 필요한 쪽은 `body_of`로 파일에서 읽는다.
    pub abspath: Option<String>,
}

/// 문서 본문 — `abspath`가 있으면 원본 파일에서 읽고, 없으면 저장된 `body`.
///
/// 파일이 사라졌거나 읽을 수 없으면 Ok(None). 인덱스가 워킹트리보다 최신일 수
/// 있으므로(브랜치 전환, 파일 삭제) 이건 에러가 아니라 정상적인 결과다.
pub fn body_of(doc: &Doc) -> Option<String> {
    match &doc.abspath {
        Some(p) => std::fs::read_to_string(p).ok(),
        None => Some(doc.body.clone()),
    }
}

impl Store {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "busy_timeout", 30000)?;
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS documents (
                id          INTEGER PRIMARY KEY,
                collection  TEXT NOT NULL,
                relpath     TEXT NOT NULL,
                title       TEXT NOT NULL DEFAULT '',
                body        TEXT NOT NULL DEFAULT '',
                context     TEXT,
                mtime_ns    INTEGER NOT NULL DEFAULT 0,
                size        INTEGER NOT NULL DEFAULT 0,
                hash        TEXT NOT NULL DEFAULT '',
                active      INTEGER NOT NULL DEFAULT 1,
                dirty       INTEGER NOT NULL DEFAULT 1,
                abspath     TEXT,
                UNIQUE(collection, relpath)
            );
            CREATE INDEX IF NOT EXISTS idx_documents_dirty ON documents(dirty) WHERE dirty = 1;
            "#,
        )?;
        // 기존 DB 마이그레이션 — 이미 있으면 duplicate column 에러를 무시한다.
        let _ = conn.execute("ALTER TABLE documents ADD COLUMN abspath TEXT", []);
        Ok(Store { conn })
    }

    /// 스캔 시 파일 메타가 기존과 같으면 skip. 다르면 upsert + dirty 마킹.
    #[allow(clippy::too_many_arguments)]
    pub fn upsert_doc(
        &mut self,
        collection: &str,
        relpath: &str,
        title: &str,
        body: &str,
        context: Option<&str>,
        mtime_ns: i64,
        size: i64,
        hash: &str,
        abspath: Option<&str>,
    ) -> Result<UpsertOutcome> {
        let existing: Option<(i64, String)> = self
            .conn
            .query_row(
                "SELECT id, hash FROM documents WHERE collection = ?1 AND relpath = ?2",
                params![collection, relpath],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .map(Some)
            .or_else(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Ok(None),
                other => Err(other),
            })?;

        match existing {
            Some((id, old_hash)) if old_hash == hash => {
                // 내용 동일 — active 유지만 보장
                self.conn.execute(
                    "UPDATE documents SET active = 1, mtime_ns = ?2, size = ?3 WHERE id = ?1",
                    params![id, mtime_ns, size],
                )?;
                Ok(UpsertOutcome::Unchanged)
            }
            Some((id, _)) => {
                self.conn.execute(
                    "UPDATE documents SET title=?2, body=?3, context=?4, mtime_ns=?5, size=?6, hash=?7, abspath=?8, active=1, dirty=1 WHERE id=?1",
                    params![id, title, body, context, mtime_ns, size, hash, abspath],
                )?;
                Ok(UpsertOutcome::Updated)
            }
            None => {
                self.conn.execute(
                    "INSERT INTO documents (collection, relpath, title, body, context, mtime_ns, size, hash, abspath, active, dirty) \
                     VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,1,1)",
                    params![collection, relpath, title, body, context, mtime_ns, size, hash, abspath],
                )?;
                Ok(UpsertOutcome::Added)
            }
        }
    }

    /// 스캔에서 보이지 않은 파일을 비활성화. 비활성화된 수 반환.
    pub fn deactivate_missing(&mut self, collection: &str, seen: &[String]) -> Result<usize> {
        let tx = self.conn.transaction()?;
        tx.execute_batch(
            "CREATE TEMP TABLE IF NOT EXISTS seen_paths (relpath TEXT PRIMARY KEY); DELETE FROM seen_paths;",
        )?;
        {
            let mut ins = tx.prepare("INSERT OR IGNORE INTO seen_paths (relpath) VALUES (?1)")?;
            for p in seen {
                ins.execute(params![p])?;
            }
        }
        let n = tx.execute(
            "UPDATE documents SET active = 0, dirty = 1 \
             WHERE collection = ?1 AND active = 1 \
               AND relpath NOT IN (SELECT relpath FROM seen_paths)",
            params![collection],
        )?;
        tx.execute_batch("DELETE FROM seen_paths;")?;
        tx.commit()?;
        Ok(n)
    }

    pub fn dirty_docs(&self) -> Result<Vec<Doc>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, collection, relpath, title, body, context, abspath FROM documents WHERE dirty = 1",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(Doc {
                id: r.get(0)?,
                collection: r.get(1)?,
                relpath: r.get(2)?,
                title: r.get(3)?,
                body: r.get(4)?,
                context: r.get(5)?,
                abspath: r.get(6)?,
            })
        })?;
        Ok(rows.collect::<std::result::Result<_, _>>()?)
    }

    pub fn is_active(&self, id: i64) -> Result<bool> {
        Ok(self.conn.query_row(
            "SELECT active FROM documents WHERE id = ?1",
            params![id],
            |r| r.get::<_, i64>(0),
        )? != 0)
    }

    pub fn clear_dirty(&mut self, ids: &[i64]) -> Result<()> {
        let tx = self.conn.transaction()?;
        {
            let mut stmt = tx.prepare("UPDATE documents SET dirty = 0 WHERE id = ?1")?;
            for id in ids {
                stmt.execute(params![id])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn counts(&self) -> Result<(i64, i64)> {
        let total: i64 =
            self.conn
                .query_row("SELECT COUNT(*) FROM documents WHERE active = 1", [], |r| {
                    r.get(0)
                })?;
        let dirty: i64 =
            self.conn
                .query_row("SELECT COUNT(*) FROM documents WHERE dirty = 1", [], |r| {
                    r.get(0)
                })?;
        Ok((total, dirty))
    }

    pub fn collection_counts(&self) -> Result<Vec<(String, i64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT collection, COUNT(*) FROM documents WHERE active = 1 GROUP BY collection ORDER BY collection",
        )?;
        let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
        Ok(rows.collect::<std::result::Result<_, _>>()?)
    }
}

#[derive(Debug, PartialEq)]
pub enum UpsertOutcome {
    Added,
    Updated,
    Unchanged,
}
