import { describe, expect, it } from "vitest";
import { normalizeSupabaseUrl } from "@/lib/supabase/url";

describe("normalizeSupabaseUrl", () => {
  it.each([
    ["https://abc.supabase.co", "https://abc.supabase.co"],
    ["https://abc.supabase.co/", "https://abc.supabase.co"],
    ["https://abc.supabase.co/rest/v1/", "https://abc.supabase.co"],
    ["  https://abc.supabase.co/auth/v1  ", "https://abc.supabase.co"],
    ["http://localhost:54321/rest/v1", "http://localhost:54321"],
  ])("reduces %s to the origin", (input, expected) => {
    expect(normalizeSupabaseUrl(input)).toBe(expected);
  });

  it("returns undefined for empty input", () => {
    expect(normalizeSupabaseUrl(undefined)).toBeUndefined();
    expect(normalizeSupabaseUrl("   ")).toBeUndefined();
  });

  it("passes an unparseable value through instead of hiding the real error", () => {
    expect(normalizeSupabaseUrl("not a url")).toBe("not a url");
  });
});
