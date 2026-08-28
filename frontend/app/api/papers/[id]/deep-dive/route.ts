import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

export async function GET(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const apiUrl = process.env.BACKEND_API_URL ?? process.env.NEXT_PUBLIC_API_URL;
  if (!apiUrl) return NextResponse.json({ error: "Backend is not configured" }, { status: 503 });
  const { id } = await params;
  try {
    const response = await fetch(`${apiUrl.replace(/\/$/, "")}/papers/${encodeURIComponent(id)}/deep-dive`, {
      headers: {
        ...(process.env.API_SHARED_SECRET ? { "X-API-Key": process.env.API_SHARED_SECRET } : {}),
        ...(request.headers.get("authorization") ? { Authorization: request.headers.get("authorization")! } : {}),
      },
      cache: "no-store",
    });
    const data = await response.json().catch(() => ({ detail: "Backend returned an invalid response" }));
    return NextResponse.json(response.ok ? data : { error: data.detail ?? "Could not open the deep dive" }, { status: response.status });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "Backend unavailable" }, { status: 502 });
  }
}
