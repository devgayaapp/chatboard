from components.text.news_event import NewsEvent
from pyrogram import Client
from pyrogram.types.messages_and_media.message import Message
from components.media.media import Media
from components.image.image import Image
from components.video.video import Video
from config import TELEGRAM_API_ID, TELEGRAM_API_HASH, TEMP_DIR, TELEGRAM_SESSION_STRING
from util.sanitizers import get_sanitized_filename
import unittest.mock
import asyncio
from datetime import datetime, timedelta




def mock_input(prompt):
    # Return the desired input value here, e.g., a phone number
    return '972546332698'


class TelegramException(Exception):
    pass



TELEGRAM_CHANNEL_TOPICS = {
    "englishabuali": "israel_palastine"
}


def get_telegram_client():
    return Client(
        "my_account", 
        TELEGRAM_API_ID, 
        TELEGRAM_API_HASH,
        session_string=TELEGRAM_SESSION_STRING
    )



def sync_get_chat_time_window(chat_id, verbouse=False, limit=20, start_date=None, end_date=None):
    telegram_messages = []
    with unittest.mock.patch('builtins.input', side_effect=mock_input):
        with get_telegram_client() as app:
            msg_count = 100
            while True and msg_count > 0:
                for message in app.get_chat_history(chat_id, limit=limit, offset_date=end_date):
                    if message.date < start_date:
                        msg_count = 0
                        break
                    telegram_messages.append(message)
                    if verbouse:
                        print(message.text)
                        print(message)
                        print("---------------------")
            return telegram_messages



async def get_chat_time_window(chat_id, limit=20, start_date=None, end_date=None, logger=None):
    telegram_messages = []
    async with get_telegram_client() as app:
        msg_count = 100
        while True and msg_count > 0:
            async for message in app.get_chat_history(chat_id, limit=limit, offset_date=end_date):
                if message.date < start_date:
                    msg_count = 0
                    break
                telegram_messages.append(message)
                if logger:
                    logger.info(f'msg: {message.text}')

            # telegram_messages[-1]
            msg_count -= 1
            if len(telegram_messages) > 0:
                end_date = telegram_messages[-1].date
            else:
                break
            if logger:
                logger.info(f'extracted {len(telegram_messages)} if iteration:{msg_count}/100, next iteration end_date: {end_date} limit: {limit}')
        return telegram_messages


async def get_chat_history(chat_id, verbouse=False, limit=20, offset=0):
    telegram_messages = []
    async with get_telegram_client() as app:
        async for message in app.get_chat_history(chat_id, limit=limit, offset=offset):
            telegram_messages.append(message)
            if verbouse:
                print(message.text)
                print(message)
                print("---------------------")
        return telegram_messages
        


def get_filename(msg):
    if not msg.media:
        raise TelegramException('message does not contain media')
    telegram_chanel = msg.chat.title
    msg_id = msg.id
    if msg.media.name == 'VIDEO':
        ext = msg.video.mime_type.split('/')[1]
    elif msg.media.name == 'PHOTO':
        ext = 'jpg'
    filename = get_sanitized_filename(f'telegram_{telegram_chanel}_{msg_id}')
    return f"{filename}.{ext}"


async def download_telegram_media(msg: Message):
    file_name = get_filename(msg)
    if msg.media:
        async with get_telegram_client() as app:
            await app.download_media(msg, file_name=TEMP_DIR / file_name)
            if msg.media.name == 'VIDEO':
                video = Video(
                    filepath=TEMP_DIR / file_name,
                    width=msg.video.width,
                    height=msg.video.height,
                    duration=msg.video.duration
                )
                thumbnail_file = await app.download_media(msg.video.thumbs[0].file_id, in_memory=True)
                thumbnail = Image.from_bytes(thumbnail_file.getbuffer())
                media = Media(
                    media = video,
                    filename = file_name,
                    size=msg.video.file_size / 1000000,
                    thumbnail=thumbnail,
                    source='hasbara',
                    date=msg.date,
                    caption=msg.caption,
                    platform_id = msg.id,
                    platform_file_id = msg.video.file_id,
                    platform_type = 'telegram',
                    platform_group=msg.chat.title
                )
            elif msg.media.name == 'PHOTO':
                thumbnail = None
                if msg.photo.thumbs is not None:
                    thumbnail_file = await app.download_media(msg.photo.thumbs[0].file_id, in_memory=True)
                    thumbnail = Image.from_bytes(thumbnail_file.getbuffer())
                image = Image.from_file(str(TEMP_DIR / file_name))
                media = Media(
                    media = image,
                    filename = file_name,
                    size=msg.photo.file_size / 1000000,
                    thumbnail=thumbnail,
                    source='hasbara',
                    date=msg.date,
                    caption=msg.caption,
                    platform_id = msg.id,
                    platform_file_id = msg.photo.file_id,
                    platform_type = 'telegram',
                    platform_group=msg.chat.title
                )
            else:
                raise TelegramException(f'message does not contain media: {msg.media.name}')

            media._is_downloaded = True
            return media
    else:
        raise TelegramException('message does not contain media')



def channel_text_sanitizers(channel, text):
    if channel == 'englishabuali':
        text = text.replace('To comment, follow this link', '')
    text = text.strip()
    return text



async def extract_telegram_lake_media(event_json, bucket, channel, partition, lazy=False):
    platform = 'telegram'
    msg = event_json['msg']
    text = msg.get('caption', None)

    if text:
        text = channel_text_sanitizers(channel, text)

    metadata = None
    if msg['media'] == 'MessageMediaType.VIDEO':
        metadata = {
            'width': msg['video']['width'],
            'height': msg['video']['height'],
            'duration': msg['video']['duration'],
        }
    media = None
    thumbnail = None
    if not lazy:
        if event_json['files']['thumbnail_filename'] is not None:
                res = await asyncio.gather(*[
                    bucket.aget_media(platform, channel, partition, event_json['files']['filename'], metadata=metadata),
                    bucket.aget_media(platform, channel, partition, event_json['files']['thumbnail_filename'])
                ])
                media, thumbnail = res
        else:
            media = await bucket.aget_media(platform, channel, partition, event_json['files']['filename'], metadata=metadata)
            thumbnail = None


    media = Media(
        media=media,
        filename=event_json['files']['filename'],
        size=msg['photo']['file_size'] / 1000000 if msg['media'] == 'MessageMediaType.PHOTO' else msg['video']['file_size'] / 1000000,
        thumbnail=thumbnail,
        source=f"telegram_{channel}",
        date=msg['date'],
        caption=text,
        platform_id = msg['id'],
        platform_file_id = msg['photo']['file_id'] if msg['media'] == 'MessageMediaType.PHOTO' else msg['video']['file_id'],
        platform_type = platform,
        platform_group=channel
    )
    media._is_downloaded = True
    return media


async def extract_telegram_lake_news_event(event_json, bucket, channel, partition, lazy=False):
    msg = event_json['msg']
    text = None
    media = None
    if 'text' in msg:
        text = msg['text']
    elif 'caption' in msg:
        text = msg['caption']
    if text:
        text = channel_text_sanitizers(channel, text)
    if 'media' in msg:
        media = await extract_telegram_lake_media(event_json, bucket, channel, partition, lazy=lazy)
    event = NewsEvent(
        date=datetime.strptime(msg['date'], "%Y-%m-%d %H:%M:%S"),
        text=text,
        media=media,
        platform_id=msg['id'],
        platform='telegram',
        channel=channel,
        topic=TELEGRAM_CHANNEL_TOPICS[channel],
    )

    return event

