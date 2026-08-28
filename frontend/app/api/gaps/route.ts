import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

export async function GET(request: NextRequest) {
  const apiUrl = process.env.BACKEND_API_URL ?? process.env.NEXT_PUBLIC_API_URL;
  if (!apiUrl) return NextResponse.json({ error: "Backend is not configured" }, { status: 503 });
  const limit = request.nextUrl.searchParams.get("limit") ?? "3";
  const sessionId = request.nextUrl.searchParams.get("session_id");
  const params = new URLSearchParams({ limit });
  if (sessionId) params.set("session_id", sessionId);
  try {
    const response = await fetch(`${apiUrl.replace(/\/$/, "")}/gaps?${params.toString()}`, {
      headers: {
        ...(process.env.API_SHARED_SECRET ? { "X-API-Key": process.env.API_SHARED_SECRET } : {}),
        ...(request.headers.get("authorization") ? { Authorization: request.headers.get("authorization")! } : {}),
      },
      cache: "no-store",
    });
    const data = await response.json();
    return NextResponse.json(response.ok ? data : { error: data.detail ?? "Could not load suggestions" }, { status: response.status });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "Backend unavailable" }, { status: 502 });
  }
}
