/** Single source for the theme storage key and how a theme is applied to <html>. Dark is the default (no attribute); light sets data-theme="light". */
export const THEME_KEY = "ragbase-theme";

export type Theme = "light" | "dark";

export function applyTheme(theme: Theme): void {
  if (theme === "light") document.documentElement.dataset.theme = "light";
  else delete document.documentElement.dataset.theme;
}
