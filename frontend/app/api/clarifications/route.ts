import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function GET() {
  const apiUrl = process.env.BACKEND_API_URL ?? process.env.NEXT_PUBLIC_API_URL;
  if (!apiUrl) return NextResponse.json({ error: "Backend is not configured" }, { status: 503 });
  try {
    const response = await fetch(`${apiUrl.replace(/\/$/, "")}/clarifications`, {
      headers: process.env.API_SHARED_SECRET ? { "X-API-Key": process.env.API_SHARED_SECRET } : {},
      cache: "no-store",
    });
    const data = await response.json();
    return NextResponse.json(response.ok ? data : { error: data.detail ?? "Could not load clarifications" }, { status: response.status });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "Backend unavailable" }, { status: 502 });
  }
}
