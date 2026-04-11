export type ChunkRecord = {
  id: string;
  text: string;
  index: number;
  charStart: number;
  length: number;
  wordCount: number;
};

export function chunkText(
  text: string,
  { chunkSize = 500, overlap = 100 }: { chunkSize?: number; overlap?: number } = {}
): ChunkRecord[] {
  const sentences = text.match(/[^.!?\n]+[.!?\n]+|[^.!?\n]+$/g) || [text];
  const chunks: { text: string; sentences: string[] }[] = [];
  let current = "";
  let currentSentences: string[] = [];
  for (const sentence of sentences) {
    const trimmed = sentence.trim();
    if (!trimmed) continue;
    if ((current + " " + trimmed).length > chunkSize && current.length > 0) {
      chunks.push({ text: current.trim(), sentences: [...currentSentences] });
      const overlapText = currentSentences.slice(-2).join(" ");
      if (overlapText.length <= overlap) {
        current = overlapText + " " + trimmed;
        currentSentences = [...currentSentences.slice(-2), trimmed];
      } else {
        current = trimmed;
        currentSentences = [trimmed];
      }
    } else {
      current = current ? current + " " + trimmed : trimmed;
      currentSentences.push(trimmed);
    }
  }
  if (current.trim()) chunks.push({ text: current.trim(), sentences: currentSentences });
  return chunks.map((c, i) => ({
    id: `chunk_${i}`,
    text: c.text,
    index: i,
    charStart: text.indexOf(c.text.substring(0, 30)),
    length: c.text.length,
    wordCount: c.text.split(/\s+/).length,
  }));
}
