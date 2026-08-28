import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

export async function POST(request: NextRequest) {
  const apiUrl = process.env.BACKEND_API_URL ?? process.env.NEXT_PUBLIC_API_URL;
  if (!apiUrl) return NextResponse.json({ error: "Backend is not configured" }, { status: 503 });
  const body = await request.json() as { paper_id?: string };
  if (!body.paper_id) return NextResponse.json({ error: "paper_id is required" }, { status: 400 });
  try {
    const response = await fetch(
      `${apiUrl.replace(/\/$/, "")}/papers/${encodeURIComponent(body.paper_id)}/guide`,
      {
        method: "POST",
        headers: {
          ...(process.env.API_SHARED_SECRET ? { "X-API-Key": process.env.API_SHARED_SECRET } : {}),
          ...(request.headers.get("authorization") ? { Authorization: request.headers.get("authorization")! } : {}),
        },
        cache: "no-store",
      },
    );
    const data = await response.json();
    return NextResponse.json(
      response.ok ? data : { error: data.detail ?? "Could not build the paper guide" },
      { status: response.status },
    );
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Backend unavailable" },
      { status: 502 },
    );
  }
}
