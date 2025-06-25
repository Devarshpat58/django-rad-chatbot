# Static Files Directory

This directory contains all static assets for the Django RAG Chatbot application.

## Directory Structure

```
static/
├── css/                 # Stylesheets
│   └── base.css        # Base styles and CSS variables
├── js/                  # JavaScript files
│   └── base.js         # Common utilities and functionality
├── images/              # Images, icons, and graphics
│   └── README.md       # Image guidelines
├── fonts/              # Custom fonts (if needed)
│   └── README.md       # Font guidelines
└── README.md           # This file
```

## Development vs Production

### Development
In development, static files are served by Django's development server.

### Production
- Run `python manage.py collectstatic` to gather all static files
- Static files are collected to `staticfiles/` directory
- Configure web server (nginx/Apache) to serve static files directly

## Django Settings

```python
# Current configuration
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'  # Production location
STATICFILES_DIRS = [BASE_DIR / 'static']  # Development source
```

## Usage in Templates

```django
{% load static %}

<!-- CSS -->
<link rel="stylesheet" href="{% static 'css/base.css' %}">

<!-- JavaScript -->
<script src="{% static 'js/base.js' %}"></script>

<!-- Images -->
<img src="{% static 'images/logo.png' %}" alt="Logo">

<!-- Fonts -->
<link rel="preload" href="{% static 'fonts/custom.woff2' %}" as="font" type="font/woff2" crossorigin>
```

## File Organization Guidelines

### CSS Files
- `base.css`: Core styles, variables, and utilities
- `components.css`: Reusable component styles
- `pages.css`: Page-specific styles
- `vendor.css`: Third-party CSS (if not using CDN)

### JavaScript Files
- `base.js`: Common utilities and functionality
- `components.js`: Interactive component code
- `pages.js`: Page-specific JavaScript
- `vendor.js`: Third-party JavaScript (if not using CDN)

### Performance Best Practices

1. **Minification**: Use minified versions in production
2. **Compression**: Enable gzip/brotli compression on web server
3. **Caching**: Set appropriate cache headers for static files
4. **CDN**: Consider using a CDN for better global performance

### Version Control
- Include source files in version control
- Exclude compiled/minified files if they're generated automatically
- Use `.gitignore` appropriately

## Current Assets

### CSS
- `base.css`: Complete base styling system with CSS variables, responsive design, and component styles

### JavaScript
- `base.js`: Utility functions including CSRF handling, notifications, loading states, and common helpers

### Images
- Directory structure ready for icons, backgrounds, logos, and illustrations

### Fonts
- Currently using system font stack for optimal performance
- Directory ready for custom fonts if needed

## Adding New Assets

1. Place files in appropriate subdirectories
2. Update relevant README files
3. Test in both development and production environments
4. Consider performance impact
5. Update documentation if needed

## Troubleshooting

### Static Files Not Loading
1. Check `DEBUG = True` in development
2. Verify `STATICFILES_DIRS` setting
3. Run `python manage.py collectstatic` for production
4. Check web server static file configuration

### Performance Issues
1. Use browser dev tools to check file sizes
2. Enable compression on web server
3. Consider using a CDN
4. Optimize images and minimize CSS/JS
