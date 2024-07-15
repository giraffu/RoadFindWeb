// routefit-chart-app/vue.config.js

module.exports = {
  devServer: {
    proxy: {
      '/upload': {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    }
  }
};
