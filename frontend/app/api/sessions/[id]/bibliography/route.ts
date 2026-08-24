import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

export async function GET(_request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const apiUrl = process.env.BACKEND_API_URL ?? process.env.NEXT_PUBLIC_API_URL;
  if (!apiUrl) return NextResponse.json({ error: "Backend is not configured" }, { status: 503 });
  const { id } = await params;
  try {
    const response = await fetch(`${apiUrl.replace(/\/$/, "")}/sessions/${encodeURIComponent(id)}/bibliography`, {
      headers: process.env.API_SHARED_SECRET ? { "X-API-Key": process.env.API_SHARED_SECRET } : {},
      cache: "no-store",
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({})) as { detail?: string };
      return NextResponse.json({ error: data.detail ?? "Could not export citations" }, { status: response.status });
    }
    const text = await response.text();
    return new NextResponse(text, { status: 200, headers: { "Content-Type": "application/x-bibtex" } });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "Backend unavailable" }, { status: 502 });
  }
}
