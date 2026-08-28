import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

export async function POST(request: NextRequest) {
  const apiUrl = process.env.BACKEND_API_URL ?? process.env.NEXT_PUBLIC_API_URL;
  if (!apiUrl) return NextResponse.json({ error: "Backend is not configured" }, { status: 503 });
  try {
    const response = await fetch(`${apiUrl.replace(/\/$/, "")}/query/feedback`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(process.env.API_SHARED_SECRET ? { "X-API-Key": process.env.API_SHARED_SECRET } : {}),
        ...(request.headers.get("authorization") ? { Authorization: request.headers.get("authorization")! } : {}),
      },
      body: await request.text(),
      cache: "no-store",
    });
    if (response.status === 204) return new NextResponse(null, { status: 204 });
    const data = await response.json().catch(() => ({ detail: "Backend returned an invalid response" }));
    return NextResponse.json({ error: data.detail ?? "Could not record feedback" }, { status: response.status });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "Backend unavailable" }, { status: 502 });
  }
}
