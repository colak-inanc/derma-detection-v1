<template>
  <div>
    <button @click="selectType('image')">Görsel Yükle</button>
    <button @click="selectType('pdf')">Tahlil Sonucu Yükle</button>
    <input type="file" ref="fileInput" @change="uploadFile" style="display:none" />
    <div v-if="loading">Yükleniyor...</div>
    <div v-if="result">{{ result }}</div>
    <div v-if="error" style="color:red">{{ error }}</div>
  </div>
</template>

<script>
export default {
  data() {
    return { fileType: '', result: '', error: '', loading: false }
  },
  methods: {
    selectType(type) {
      this.fileType = type;
      this.$refs.fileInput.click();
    },
    async uploadFile(e) {
      this.result = '';
      this.error = '';
      this.loading = true;
      const file = e.target.files[0];
      const formData = new FormData();
      formData.append('file', file);
      formData.append('type', this.fileType);
      try {
        const res = await fetch('/upload', { method: 'POST', body: formData });
        const data = await res.json();
        this.result = data.result || '';
        this.error = data.error || '';
      } catch (err) {
        this.error = 'Sunucuya bağlanılamadı!';
      }
      this.loading = false;
    }
  }
}
</script>
