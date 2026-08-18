import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

interface AdkPart { text?: string }
interface AdkEvent { author?: string; content?: { role?: string; parts?: AdkPart[] } }

export async function POST(request: NextRequest) {
  const adkUrl = process.env.ADK_API_URL ?? process.env.NEXT_PUBLIC_API_URL;
  if (!adkUrl) {
    return NextResponse.json({ error: "ADK_API_URL is not configured" }, { status: 503 });
  }

  const body = await request.json() as { message?: string; sessionId?: string };
  const message = body.message?.trim();
  if (!message) return NextResponse.json({ error: "message is required" }, { status: 400 });

  const base = adkUrl.replace(/\/$/, "");
  const appName = process.env.ADK_APP_NAME ?? "hello_world";
  const userId = `atlas-web-${(body.sessionId ?? "session").replace(/[^a-zA-Z0-9_-]/g, "").slice(0, 80)}`;
  const sessionId = body.sessionId?.replace(/[^a-zA-Z0-9_-]/g, "").slice(0, 100) || "default";

  try {
    const create = await fetch(`${base}/apps/${encodeURIComponent(appName)}/users/${encodeURIComponent(userId)}/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    });
    if (!create.ok && create.status !== 409) throw new Error(`Could not create ADK session (${create.status})`);

    const run = await fetch(`${base}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        app_name: appName,
        user_id: userId,
        session_id: sessionId,
        new_message: { role: "user", parts: [{ text: message }] },
        streaming: false,
      }),
    });
    if (!run.ok) {
      const detail = await run.text();
      throw new Error(`Gemini agent failed (${run.status}): ${detail.slice(0, 300)}`);
    }

    const events = await run.json() as AdkEvent[];
    const answer = events
      .filter((event) => event.content?.role === "model" || event.author)
      .flatMap((event) => event.content?.parts ?? [])
      .map((part) => part.text?.trim())
      .filter((text): text is string => Boolean(text))
      .join("\n\n")
      .trim();
    if (!answer) throw new Error("Gemini returned no text");
    return NextResponse.json({ answer, citations: [], retrieval_mode: "no_results" });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "Gemini agent unavailable" }, { status: 502 });
  }
}
