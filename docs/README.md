# Ooblex Documentation Website

This folder contains the public-facing website for Ooblex, showcasing the platform's features and capabilities.

## Pages

- `index.html` - Main landing page highlighting key features and benefits
- `gettingstarted.html` - Quick start guide for deploying Ooblex with Docker
- `features.html` - Comprehensive feature showcase

## Serving Locally

To view the website locally:

```bash
# Using Python
python -m http.server 8000

# Using Node.js
npx http-server

# Using Docker
docker run -p 8000:80 -v $(pwd):/usr/share/nginx/html nginx
```

Then visit http://localhost:8000

## Deployment

The website can be deployed to any static hosting service:
- GitHub Pages
- Netlify
- Vercel
- AWS S3 + CloudFront
- Cloudflare Pages

## Modern Features Highlighted

The updated website showcases all of Ooblex's modern capabilities:

- **WebRTC Streaming**: WHIP/WHEP protocols, sub-100ms latency
- **Edge Computing**: WebAssembly modules for browser-based AI
- **Blockchain Trust**: Content authenticity and deepfake detection
- **Mobile SDKs**: Native iOS/Android, Flutter, React Native
- **Real-time Collaboration**: Multi-user synchronized sessions
- **Cloud Native**: Kubernetes, Docker, Helm charts
- **Enterprise Ready**: Monitoring, security, scalability

## Contributing

When updating the website, ensure all new features are properly documented and the messaging remains consistent across all pages.