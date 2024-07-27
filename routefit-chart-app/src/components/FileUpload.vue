<!--
 * @Author: chuzeyu 3343447088@qq.com
 * @Date: 2024-06-28 10:43:04
 * @LastEditors: chuzeyu 3343447088@qq.com
 * @LastEditTime: 2024-07-22 10:27:41
 * @FilePath: \routefit-chart-app\src\components\FileUpload.vue
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
<!--
 * @Author: chuzeyu 3343447088@qq.com
 * @Date: 2024-06-28 10:43:04
 * @LastEditors: chuzeyu 3343447088@qq.com
 * @LastEditTime: 2024-07-20 10:01:52
 * @FilePath: \RoadFindWeb\routefit-chart-app\src\components\FileUpload.vue
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
<template>
  <div>
    <h1>上传 CSV 或 XLSX 文件</h1>
    <input type="file" @change="handleFileUpload" accept=".csv,.xlsx" />
    <button @click="submitFile">提交</button>
    <button @click="calculateData" :disabled="!filePath">计算</button>
    <PlotlyChart :chartData="chartData" :key="chartKey" />
    <div v-if="chartData" class="controls-container">
      <div
        v-for="(param, index) in initialParams"
        :key="index"
        class="control-group"
      >
        <label>第{{ index + 1 }}段圆曲线中心点位置:</label>
        <input v-model="centers[index]" @input="updateChart" type="number" />

        <label>第{{ index + 1 }}段圆曲线长度:</label>
        <input v-model="lengths[index]" @input="updateChart" type="number" />

        <label>第{{ index + 1 }}段回旋线类型:</label>
        <select v-model="types[index]" @change="updateChart">
          <option value="基本对称">基本对称</option>
          <option value="基本非对称">基本非对称</option>
          <option value="S型">S型</option>
          <option value="卵型">卵型</option>
          <option value="凸型">凸型</option>
          <option value="C型">C型</option>
          <option value="复合型">复合型</option>
        </select>
      </div>
    </div>
  </div>
</template>

<script>
import axios from "axios";
import PlotlyChart from "./PlotlyChart.vue";

export default {
  components: {
    PlotlyChart,
  },
  data() {
    return {
      file: null,
      filePath: null,
      chartData: null,
      initialParams: [],
      centers: [],
      lengths: [],
      tolerances:[],
      types: [],
      chartKey: 0, // 添加一个 key 用于重新渲染 PlotlyChart
    };
  },
  methods: {
    handleFileUpload(event) {
      this.file = event.target.files[0];
      console.log("File selected:", this.file); // 调试信息
    },
    async submitFile() {
      if (!this.file) return;

      const formData = new FormData();
      formData.append("file", this.file);
      console.log("Submitting file:", this.file); // 调试信息

      try {
        const response = await axios.post(
          "http://10.137.118.157:8000/route/upload/",
          formData,
          {
            headers: {
              "Content-Type": "multipart/form-data",
            },
          }
        );

        console.log("Upload response:", response.data); // 调试信息
        this.filePath = response.data.file_path;
        console.log("File path:", this.filePath); // 调试信息
      } catch (error) {
        console.error("Error uploading file:", error);
      }
    },
    async calculateData() {
      if (!this.filePath) return;

      const formData = new FormData();
      formData.append("file_path", this.filePath);

      try {
        console.log("Sending file path for calculation:", this.filePath); // 调试信息
        const response = await axios.post(
          "http://10.137.118.157:8000/route/calculate/",
          formData,
          {
            headers: {
              "Content-Type": "multipart/form-data",
            },
            
          }
        );

        console.log("Calculation response:", response.data); // 调试信息
        this.chartData = response.data;
        this.initialParams = response.data.tolerances;
        this.centers = response.data.centers; // 设置默认中心点位置
        this.tolerances = response.data.tolerances;//圆曲线容差
        this.lengths = [...this.initialParams]; // 使用 initial_params 初始化长度数组
        this.types = new Array(this.initialParams.length).fill("基本对称"); // 设置默认回旋线类型
        this.renderChart();
      } catch (error) {
        console.error("Error calculating data:", error);
      }
    },

    async getData() {
      if (!this.filePath) return;

      const formData = new FormData();
      formData.append("file_path", this.filePath);

      try {
        console.log("Sending file path for calculation:", this.filePath); // 调试信息
        const response = await axios.post(
          "http://10.137.118.157:8000/route/calculate/",
          formData,
          {
            headers: {
              "Content-Type": "multipart/form-data",
            },
          }
        );

        console.log("Calculation response:", response.data); // 调试信息
        this.chartData = response.data;
        this.initialParams = response.data.tolerances;
        this.centers = response.data.centers; // 设置默认中心点位置
        this.tolerances = response.data.tolerances;//圆曲线容差
        this.lengths = [...this.initialParams]; // 使用 initial_params 初始化长度数组
        this.types = new Array(this.initialParams.length).fill("基本对称"); // 设置默认回旋线类型
        this.renderChart();
      } catch (error) {
        console.error("Error calculating data:", error);
      }
    },
    async updateChart() {
      // 重新计算的逻辑
      const recalculationData = {
        centers: this.centers,
        lengths: this.lengths,
        types: this.types,
      };

      try {
        const response = await axios.post(
          "http://10.137.118.157:8000/route/recalculate/",
          recalculationData,
          {
            headers: {
              "Content-Type": "application/json",
            },
          }
        );

        console.log("Recalculation response:", response.data); // 调试信息
        this.chartData = response.data;
        // 增加 chartKey 来强制子组件重新渲染
        this.chartKey += 1;
        this.renderChart();
      } catch (error) {
        console.error("Error recalculating data:", error);
      }
    },

    renderChart() {
      if (!this.chartData) return;

      //const ctx = document.getElementById('splineChart').getContext('2d');
      // Your chart rendering logic here
    },
  },
};
</script>

<style>
.controls-container {
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
}

.control-group {
  display: flex;
  flex-direction: column;
  margin-bottom: 20px;
}

.control-group label {
  margin-bottom: 5px;
}

.control-group input,
.control-group select {
  margin-bottom: 10px;
}
</style>
