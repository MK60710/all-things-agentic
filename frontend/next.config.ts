import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Next 16 blocks dev-only client bundles when localhost is opened via its
  // numeric loopback alias unless that host is explicitly allowed. Without
  // this, the HTML renders but React never hydrates, leaving every control inert.
  allowedDevOrigins: ["127.0.0.1"],
};

export default nextConfig;
