import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

export async function GET(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const apiUrl = process.env.BACKEND_API_URL ?? process.env.NEXT_PUBLIC_API_URL;
  if (!apiUrl) return NextResponse.json({ error: "Backend is not configured" }, { status: 503 });
  const { id } = await params;
  try {
    const response = await fetch(`${apiUrl.replace(/\/$/, "")}/sessions/${encodeURIComponent(id)}/messages`, {
      headers: {
        ...(process.env.API_SHARED_SECRET ? { "X-API-Key": process.env.API_SHARED_SECRET } : {}),
        ...(request.headers.get("authorization") ? { Authorization: request.headers.get("authorization")! } : {}),
      },
      cache: "no-store",
    });
    const data = await response.json();
    return NextResponse.json(response.ok ? data : { error: data.detail ?? "Could not load messages" }, { status: response.status });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "Backend unavailable" }, { status: 502 });
  }
}

export async function PUT(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const apiUrl = process.env.BACKEND_API_URL ?? process.env.NEXT_PUBLIC_API_URL;
  if (!apiUrl) return NextResponse.json({ error: "Backend is not configured" }, { status: 503 });
  const { id } = await params;
  try {
    const response = await fetch(`${apiUrl.replace(/\/$/, "")}/sessions/${encodeURIComponent(id)}/messages`, {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
        ...(process.env.API_SHARED_SECRET ? { "X-API-Key": process.env.API_SHARED_SECRET } : {}),
        ...(request.headers.get("authorization") ? { Authorization: request.headers.get("authorization")! } : {}),
      },
      body: await request.text(),
      cache: "no-store",
    });
    const data = await response.json();
    return NextResponse.json(response.ok ? data : { error: data.detail ?? "Could not save messages" }, { status: response.status });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "Backend unavailable" }, { status: 502 });
  }
}
