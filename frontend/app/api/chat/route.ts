import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

export async function POST(request: NextRequest) {
  const apiUrl = process.env.BACKEND_API_URL ?? process.env.NEXT_PUBLIC_API_URL;
  if (!apiUrl) {
    return NextResponse.json({ error: "BACKEND_API_URL is not configured" }, { status: 503 });
  }
  const body = await request.json();
  try {
    const response = await fetch(`${apiUrl.replace(/\/$/, "")}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(process.env.API_SHARED_SECRET ? { "X-API-Key": process.env.API_SHARED_SECRET } : {}),
        ...(request.headers.get("authorization") ? { Authorization: request.headers.get("authorization")! } : {}),
      },
      body: JSON.stringify(body),
      cache: "no-store",
    });
    const data = await response.json().catch(() => ({ detail: "Backend returned an invalid response" }));
    if (!response.ok) {
      const outgoing = NextResponse.json(
        { error: data.detail ?? `Backend request failed (${response.status})` },
        { status: response.status },
      );
      const retryAfter = response.headers.get("retry-after");
      if (retryAfter) outgoing.headers.set("Retry-After", retryAfter);
      return outgoing;
    }
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Backend unavailable" },
      { status: 502 },
    );
  }
}
