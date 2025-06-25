# Static Fonts Directory

This directory contains custom fonts for the Django RAG Chatbot.

## Font Guidelines

### Web Font Formats
- **WOFF2**: Modern compression, preferred format
- **WOFF**: Fallback for older browsers
- **TTF/OTF**: System fonts (if needed)

### Font Loading Strategy
```css
@font-face {
    font-family: 'CustomFont';
    src: url('../fonts/customfont.woff2') format('woff2'),
         url('../fonts/customfont.woff') format('woff');
    font-display: swap; /* Improves loading performance */
    font-weight: 400;
    font-style: normal;
}
```

### Performance Considerations
- Use `font-display: swap` for better loading experience
- Preload critical fonts in HTML head:
  ```html
  <link rel="preload" href="{% static 'fonts/main-font.woff2' %}" as="font" type="font/woff2" crossorigin>
  ```
- Limit the number of font weights and styles
- Consider using system font stacks for better performance:
  ```css
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
  ```

## Usage in CSS

```css
body {
    font-family: 'CustomFont', -apple-system, BlinkMacSystemFont, sans-serif;
}
```

## License Considerations
- Ensure all fonts are properly licensed for web use
- Keep license files alongside font files
- Document font sources and licensing terms

## Current Fonts

*Add descriptions of fonts as they are added to the project*

### System Font Stack (Currently Used)
```css
font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
```

This provides excellent cross-platform consistency without additional downloads.
