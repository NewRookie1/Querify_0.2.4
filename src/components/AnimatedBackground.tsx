"use client";

export default function AnimatedBackground({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="relative min-h-screen overflow-hidden bg-black">
      {/* Aurora Gradient */}
      <div className="absolute inset-0 -z-10 animate-aurora bg-[radial-gradient(circle_at_20%_20%,#a855f7,transparent_40%),radial-gradient(circle_at_80%_30%,#6366f1,transparent_40%),radial-gradient(circle_at_50%_80%,#22d3ee,transparent_40%)] opacity-70" />

      {/* Floating Blobs */}
      <div className="absolute -top-24 -left-24 h-96 w-96 rounded-full bg-purple-500 opacity-30 blur-3xl animate-float" />
      <div className="absolute top-1/2 -right-24 h-96 w-96 rounded-full bg-cyan-400 opacity-30 blur-3xl animate-float-delayed" />

      {children}
    </div>
  );
}
