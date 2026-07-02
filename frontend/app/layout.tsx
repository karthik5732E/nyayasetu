import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Nyaya Setu — Offline AI Legal Assistant",
  description: "Private, offline AI legal assistant for Indian law.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}