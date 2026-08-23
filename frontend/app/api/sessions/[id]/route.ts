import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

export async function PATCH(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const apiUrl = process.env.BACKEND_API_URL ?? process.env.NEXT_PUBLIC_API_URL;
  if (!apiUrl) return NextResponse.json({ error: "Backend is not configured" }, { status: 503 });
  const { id } = await params;
  try {
    const response = await fetch(`${apiUrl.replace(/\/$/, "")}/sessions/${encodeURIComponent(id)}`, {
      method: "PATCH",
      headers: {
        "Content-Type": "application/json",
        ...(process.env.API_SHARED_SECRET ? { "X-API-Key": process.env.API_SHARED_SECRET } : {}),
      },
      body: await request.text(),
      cache: "no-store",
    });
    const data = await response.json();
    return NextResponse.json(response.ok ? data : { error: data.detail ?? "Could not rename the session" }, { status: response.status });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "Backend unavailable" }, { status: 502 });
  }
}

export async function DELETE(_request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const apiUrl = process.env.BACKEND_API_URL ?? process.env.NEXT_PUBLIC_API_URL;
  if (!apiUrl) return NextResponse.json({ error: "Backend is not configured" }, { status: 503 });
  const { id } = await params;
  try {
    const response = await fetch(`${apiUrl.replace(/\/$/, "")}/sessions/${encodeURIComponent(id)}`, {
      method: "DELETE",
      headers: process.env.API_SHARED_SECRET ? { "X-API-Key": process.env.API_SHARED_SECRET } : {},
      cache: "no-store",
    });
    // Backend returns 204 with no body on success - unlike the other
    // proxy routes here, there's nothing to JSON-parse in that case.
    if (response.status === 204) return new NextResponse(null, { status: 204 });
    const data = await response.json();
    return NextResponse.json({ error: data.detail ?? "Could not delete the session" }, { status: response.status });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "Backend unavailable" }, { status: 502 });
  }
}
