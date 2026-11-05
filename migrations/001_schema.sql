CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id      uuid NOT NULL,
  source       text NOT NULL,        -- job_post | email | note | interview
  source_ref   text,                 -- URL or message id
  title        text,
  content_hash text NOT NULL,        -- sha256 of normalized text: exact-dup guard
  metadata     jsonb NOT NULL DEFAULT '{}',
  created_at   timestamptz NOT NULL DEFAULT now(),
  UNIQUE (user_id, content_hash)
);

CREATE TABLE chunks (
  id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id          uuid NOT NULL,
  document_id      uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  ord              int  NOT NULL,
  text             text NOT NULL,
  embedding        vector(384) NOT NULL,   -- match your embedding model's size
  tsv              tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
  importance       real NOT NULL DEFAULT 0.5,
  last_accessed_at timestamptz,
  superseded_by    uuid REFERENCES chunks(id),
  created_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX chunks_vec ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_tsv ON chunks USING gin (tsv);
CREATE INDEX chunks_user ON chunks (user_id);
CREATE INDEX chunks_document ON chunks (document_id);
