from rest_framework import serializers

class FileUploadSerializer(serializers.Serializer):
    file = serializers.FileField()

class FilePathSerializer(serializers.Serializer):
    file_path = serializers.CharField()

class DataSerializer(serializers.Serializer):
    centers = serializers.ListField(child=serializers.IntegerField())
    lengths = serializers.ListField(child=serializers.FloatField())
    file_path = serializers.CharField()
