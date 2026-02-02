from azure.storage.blob import BlobServiceClient, ContentSettings
from io import BytesIO
from PIL import Image
import uuid
from config import AZURE_BLOB_CONNECTION_STRING, AZURE_BLOB_CONTAINER_NAME

class BlobStorageUploader:
    def __init__(self):
        self.client = BlobServiceClient.from_connection_string(
            AZURE_BLOB_CONNECTION_STRING
        )
        self.container = self.client.get_container_client(
            AZURE_BLOB_CONTAINER_NAME
        )

    def upload(self, pil_image):
        blob_name = f"{uuid.uuid4()}.png"

        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)

        blob = self.container.get_blob_client(blob_name)

        blob.upload_blob(
            buffer.getvalue(),
            overwrite=True,
            content_settings=ContentSettings(content_type="image/png")
        )

        return blob.url

uploader = BlobStorageUploader()
