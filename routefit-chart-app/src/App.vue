<!-- routefit-chart-app/src/App.vue -->
<template>
  <div id="app">
    <FileUpload />
  </div>
</template>

<script>
import FileUpload from './components/FileUpload.vue';

export default {
  name: 'App',
  components: {
    FileUpload
  }
};
</script>

<style>
/* 添加一些基本样式 */
</style>
