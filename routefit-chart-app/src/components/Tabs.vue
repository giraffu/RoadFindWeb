<template>
  <div class="tabs">
    <div class="tab-buttons">
      <button 
        v-for="(tab, index) in tabs" 
        :key="index" 
        :class="{ active: activeTab === index }"
        @click="selectTab(index)">
        {{ tab }}
      </button>
    </div>
    <div class="tab-content">
      <slot :name="`tab-${activeTab}`"></slot>
    </div>
  </div>
</template>

<script>
export default {
  name: 'TabsComponent',
  props: {
    tabs: {
      type: Array,
      required: true
    }
  },
  data() {
    return {
      activeTab: 0
    };
  },
  methods: {
    selectTab(index) {
      this.activeTab = index;
    }
  }
};
</script>

<style scoped>
.tabs {
  display: flex;
  flex-direction: column;
}

.tab-buttons {
  display: flex;
  margin-bottom: 1rem;
}

.tab-buttons button {
  flex: 1;
  padding: 1rem;
  border: none;
  background-color: #f0f0f0;
  cursor: pointer;
  transition: background-color 0.3s;
}

.tab-buttons button.active {
  background-color: #ddd;
}

.tab-buttons button:not(:last-child) {
  border-right: 1px solid #ccc;
}

.tab-content {
  padding: 1rem;
  border: 1px solid #ccc;
}
</style>
