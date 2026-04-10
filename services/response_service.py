from models.schemas import ProductionResponse, EventLogEntry, ProcessStatus, MediaInfo
from services.mongodb import mongo_handler
from datetime import datetime
import uuid
from typing import Dict, Any
import asyncio

async def create_production_response(
    user_id: str,
    feature_name: str,
    input_url: str,
    report_data: Dict[str, Any],
    start_time: datetime,
    request_type: str = "URL"
) -> ProductionResponse:
    end_time = datetime.now()
    request_id = f"MFM_{int(end_time.timestamp())}_{user_id}_{request_type}_{uuid.uuid4().hex[:5]}"
    
    event_log = {
        feature_name: EventLogEntry(
            webFeatureKey=feature_name,
            eventId=f"{user_id}_{feature_name}_{int(end_time.timestamp())}",
            processingStartTime=start_time.isoformat(),
            processingEndTime=end_time.isoformat(),
            report=report_data
        )
    }
    
    process_status = ProcessStatus(
        totalProcessingFeatures=1,
        completedProcesses=1,
        inProcess=0,
        complete=True,
        featureStatus={feature_name: True}
    )
    
    media_info = MediaInfo(
        inputMediaURL=input_url,
        type="IMAGE"
    )
    
    response = ProductionResponse(
        _id=request_id,
        mediaId=request_id,
        userId=user_id,
        requestType=request_type,
        orgId=user_id,  # Defaulting orgId to userId
        eventStartTime=start_time.strftime("%Y-%m-%d %H:%M:%S"),
        eventEndTime=end_time.strftime("%Y-%m-%d %H:%M:%S"),
        eventLog=event_log,
        processStatus=process_status,
        media=media_info,
        timedOut=False,
        saved_at=end_time.isoformat()
    )
    
    # Save to MongoDB asynchronously
    asyncio.create_task(mongo_handler.save_response(user_id, response.dict(by_alias=True)))
    
    return response
