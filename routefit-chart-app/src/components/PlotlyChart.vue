<!--
 * @Author: chuzeyu 3343447088@qq.com
 * @Date: 2024-06-28 15:57:48
 * @LastEditors: chuzeyu 3343447088@qq.com
 * @LastEditTime: 2024-07-04 15:50:47
 * @FilePath: \routefit-chart-app\src\components\SplinePlot.vue
 * @Description: This is a default setup, please set `customMade`, open koroFileHeader to view configuration and set: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
<template>
  <div ref="plot"></div>
</template>

<script>
import Plotly from 'plotly.js-dist';

export default {
  props: {
    chartData: {
      type: Object,
      required: true
    }
  },
  watch: {
    chartData: {
      immediate: true,
      handler(newData) {
        this.renderChart(newData);
      }
    }
  },
  methods: {
    renderChart(data) {
      if (!data) return;
      this.$nextTick(() => {
        console.log("点集数据：", data)
        const zero_segments_x = data.zero_segments.x || [];
        const zero_segments_y = data.zero_segments.y || [];
        const non_zero_segments_x = data.non_zero_segments.x || [];
        const non_zero_segments_y = data.non_zero_segments.y || [];
        const fits  = data.fits;
        const zero_traces = zero_segments_x.map((xArray, index) => ({
          x: xArray,
          y: zero_segments_y[index],
          mode: 'lines+markers',
          name: `直线段 ${index + 1}`,
          line: { color: 'blue', width: 2 } 
        }));

        const non_zero_traces = non_zero_segments_x.map((xArray, index) => ({
          x: xArray,
          y: non_zero_segments_y[index],
          mode: 'lines+markers',
          name: `缓和曲线 ${index + 1}`,
          line: { color: 'red', width: 2 } 
        }));

        const fits_traces = fits.x.map((xArray, index) => ({
          x: xArray,
          y: fits.y[index],
          mode: 'lines+markers',
          name: `圆曲线 ${index + 1}`,
          line: { color: 'green', width: 3 } 
        }));

        const traces = [...zero_traces, ...non_zero_traces, ...fits_traces];

        const layout = {
          title: '逐桩坐标拟合平曲线',
          xaxis: { title: 'X' },
          yaxis: { title: 'Y' }
        };

        Plotly.newPlot(this.$refs.plot, traces, layout);
    });

    }
  }
};
</script>

<style scoped>
/* 可以添加一些基本样式 */
</style>
