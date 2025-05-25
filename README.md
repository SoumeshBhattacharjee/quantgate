# QUANTGATE - Advanced Stock Market Terminal

A comprehensive stock market analysis and trading platform built with Streamlit.

## Features

- Real-time stock data visualization
- Technical analysis with multiple indicators
- Live price tracking
- Portfolio management
- Global market overview
- News feed integration
- Multi-exchange support

## Deployment Instructions

### Local Development

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run terminal_new.py
   ```

### Streamlit Cloud Deployment

1. Push your code to a GitHub repository
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app"
4. Select your repository, branch, and main file (terminal_new.py)
5. Click "Deploy"

## Environment Variables

No environment variables are required for basic functionality. The application uses public APIs.

## Dependencies

- streamlit==1.32.0
- yfinance==0.2.36
- pandas==2.2.1
- plotly==5.19.0
- requests==2.31.0
- feedparser==6.0.11
- numpy==1.26.4

## Support

For issues or feature requests, please open an issue in the GitHub repository.

## License

MIT License 