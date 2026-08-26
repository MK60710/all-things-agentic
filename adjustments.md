# Future adjustments

## Broaden academic paper discovery beyond arXiv

The current search experience intentionally uses the public arXiv API. A
future version should add a provider abstraction so reading-path discovery can
search multiple scholarly indexes without changing the UI or recommendation
model.

Potential providers:

- Semantic Scholar for paper search, related papers, and citation links.
- OpenAlex for broad scholarly metadata and citation-graph data.
- Google Scholar as an external user-facing search link, or through a
  separately licensed provider if compliant programmatic access becomes
  available.

Google Scholar should not be scraped directly: its help documentation says to
respect `robots.txt` and does not offer bulk access. Search results from any
future provider should be deduplicated, labeled with their source, and only
presented as verified recommendations when their metadata can be resolved.

The existing arXiv search and ingestion flow should remain available as the
default until this multi-source discovery layer is implemented.
