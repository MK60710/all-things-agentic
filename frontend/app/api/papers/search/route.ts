import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

function decodeXml(value: string) {
  return value
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/\s+/g, " ")
    .trim();
}

function field(entry: string, name: string) {
  return decodeXml(entry.match(new RegExp(`<${name}[^>]*>([\\s\\S]*?)<\\/${name}>`))?.[1] ?? "");
}

export async function GET(request: NextRequest) {
  const query = request.nextUrl.searchParams.get("q")?.trim();
  if (!query || query.length < 2) return NextResponse.json({ papers: [] });

  try {
    const search = encodeURIComponent(`all:${query}`);
    const response = await fetch(`https://export.arxiv.org/api/query?search_query=${search}&start=0&max_results=6&sortBy=relevance`, {
      headers: { "User-Agent": "AtlasResearchAssistant/0.1" },
      next: { revalidate: 900 },
    });
    if (!response.ok) throw new Error(`arXiv returned ${response.status}`);
    const xml = await response.text();
    const entries = xml.match(/<entry>[\s\S]*?<\/entry>/g) ?? [];
    const papers = entries.map((entry) => {
      const idUrl = field(entry, "id");
      const id = idUrl.split("/").pop()?.replace(/v\d+$/, "") ?? idUrl;
      const authors = [...entry.matchAll(/<author>[\s\S]*?<name>([\s\S]*?)<\/name>[\s\S]*?<\/author>/g)].map((match) => decodeXml(match[1])).join(", ");
      return {
        id: `arxiv:${id}`,
        title: field(entry, "title"),
        authors,
        abstract: field(entry, "summary"),
        published: field(entry, "published").slice(0, 10),
        pdfUrl: `https://arxiv.org/pdf/${id}`,
      };
    });
    return NextResponse.json({ papers });
  } catch (error) {
    return NextResponse.json({ papers: [], error: error instanceof Error ? error.message : "Search unavailable" }, { status: 502 });
  }
}
