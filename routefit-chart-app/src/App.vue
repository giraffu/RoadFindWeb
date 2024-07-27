<template>
  <div>
    <TabsComponent :tabs="['平面拟合', '纵断面拟合', 'Tab 3']">
      <template v-slot:tab-0>
        <FileUpload />
      </template>
      <template v-slot:tab-1>
        <FileUpload_lon />
      </template>
      <template v-slot:tab-2>
        <h1>Content for Tab 3</h1>
        <p>This is the content for the third tab.</p>
      </template>
    </TabsComponent>
  </div>
</template>

<script>
import TabsComponent from './components/Tabs.vue';
import FileUpload from './components/FileUpload.vue';
import FileUpload_lon from './components/FileUpload_lon.vue';

export default {
  components: {
    TabsComponent,
    FileUpload,
    FileUpload_lon
  }
};
</script>
