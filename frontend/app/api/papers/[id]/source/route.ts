import { NextResponse } from "next/server";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const apiUrl = process.env.BACKEND_API_URL ?? process.env.NEXT_PUBLIC_API_URL;
  if (!apiUrl) return NextResponse.json({ error: "Backend is not configured" }, { status: 503 });
  const { id } = await params;
  const response = await fetch(`${apiUrl.replace(/\/$/, "")}/papers/${encodeURIComponent(id)}/source`, {
    headers: process.env.API_SHARED_SECRET ? { "X-API-Key": process.env.API_SHARED_SECRET } : {},
  });
  if (!response.ok) return NextResponse.json({ error: "Paper source is not available" }, { status: response.status });
  return new NextResponse(response.body, {
    status: response.status,
    headers: {
      "Content-Type": response.headers.get("content-type") ?? "application/pdf",
      "Content-Disposition": response.headers.get("content-disposition") ?? "inline",
    },
  });
}
