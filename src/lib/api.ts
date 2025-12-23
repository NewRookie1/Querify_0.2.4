const BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL;

if (!BASE_URL) {
  throw new Error("NEXT_PUBLIC_API_BASE_URL is not defined");
}

/* ---------- Types ---------- */

export interface ChatResponse {
  reply: string;
  used_context: boolean;
  sources: Record<string, any>[];
  active_mode: string;
}

export interface PDFChatResponse {
  reply: string;
  metadata: Record<string, any>;
}

/* ---------- Normal Chat ---------- */

export async function chatRequest(
  message: string,
  mode: string
): Promise<ChatResponse> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 60_000);

  try {
    const res = await fetch(`${BASE_URL}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message, mode }),
      signal: controller.signal,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Chat request failed");
    }

    return res.json();
  } finally {
    clearTimeout(timeout);
  }
}

/* ---------- PDF Chat ---------- */

export async function chatWithPdf(
  file: File,
  message?: string
): Promise<PDFChatResponse> {
  const formData = new FormData();
  formData.append("file", file);
  if (message) {
    formData.append("message", message);
  }

  const res = await fetch(`${BASE_URL}/chat/pdf`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "PDF chat failed");
  }

  return res.json();
}
