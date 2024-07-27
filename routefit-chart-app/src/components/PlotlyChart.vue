<!--
 * @Author: chuzeyu 3343447088@qq.com
 * @Date: 2024-06-28 15:57:48
 * @LastEditors: chuzeyu 3343447088@qq.com
 * @LastEditTime: 2024-07-24 09:25:12
 * @FilePath: \routefit-chart-app\src\components\PlotlyChart.vue
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
<template>
  <div class="plot-container">
    <div ref="plot" class="plot"></div>
    <button @click="toggleAspectRatio" class="toggle-button">切换</button>
  </div>
</template>

<script>
import Plotly from "plotly.js-dist";

export default {
  props: {
    chartData: {
      type: Object,
      required: true,
    },
  },
  data() {
    return {
      isSquareAspectRatio: true, // Track the current aspect ratio state
    };
  },
  watch: {
    chartData: {
      immediate: true,
      handler(newData) {
        this.renderChart(newData);
      },
    },
  },
  methods: {
    toggleAspectRatio() {
      this.isSquareAspectRatio = !this.isSquareAspectRatio;
      this.renderChart(this.chartData); // Re-render the chart with the new aspect ratio
    },
    renderChart(data) {
      if (!data) return;
      this.$nextTick(() => {
        console.log("点集数据：", data);
        const zero_segments_x = data.zero_segments.x || [];
        const zero_segments_y = data.zero_segments.y || [];
        const non_zero_segments_x = data.non_zero_segments.x || [];
        const non_zero_segments_y = data.non_zero_segments.y || [];
        const fits = data.fits;
        const zero_traces = zero_segments_x.map((xArray, index) => ({
          x: xArray,
          y: zero_segments_y[index],
          mode: "lines+markers",
          name: `直线段 ${index + 1}`,
          line: { color: "blue", width: 2 },
        }));

        const non_zero_traces = non_zero_segments_x.map((xArray, index) => ({
          x: xArray,
          y: non_zero_segments_y[index],
          mode: "lines+markers",
          name: `缓和曲线 ${index + 1}`,
          line: { color: "red", width: 2 },
        }));

        const fits_traces = fits.x.map((xArray, index) => ({
          x: xArray,
          y: fits.y[index],
          mode: "lines+markers",
          name: `圆曲线 ${index + 1}`,
          line: { color: "green", width: 3 },
        }));

        const traces = [...zero_traces, ...non_zero_traces, ...fits_traces];

        const layout = {
          title: "逐桩坐标拟合",
          xaxis: { title: "X", scaleanchor: this.isSquareAspectRatio ? "y" : undefined },
          yaxis: { title: "Y" },
          aspectratio: this.isSquareAspectRatio ? { x: 1, y: 1 } : {},
        };

        Plotly.newPlot(this.$refs.plot, traces, layout);
      });
    },
  },
};
</script>

<style scoped>
.plot-container {
  display: flex;
  align-items: flex-start;
}

.plot {
  flex: 1;
}

.toggle-button {
  margin-left: 10px;
  align-self: flex-start; /* Align the button to the start of the container vertically */
}
</style>
