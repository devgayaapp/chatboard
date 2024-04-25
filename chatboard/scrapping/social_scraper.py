
from pymongo import MongoClient
from typing import Optional
from botocore.tokens import datetime
from bson import ObjectId
# from components.adapters.social_scraper.db.crud.post import CRUDPost
from components.text.news_event import NewsEvent
from components.adapters.social_scraper.db.crud._base import MongoCRUDBase
from components.adapters.social_scraper.db.mongodb import DataBase
# from components.adapters.social_scraper.db_models.post import Post, PostBase, PostCreate
from components.adapters.social_scraper.db_models.tw_post import Post as TwPost
from components.adapters.social_scraper.db_models.ig_post import Post as IgPost
from components.video.video import Video, get_url_video_extension
from components.image.image import Image
from components.media.media import Media
from config import SCRAPER_MONGODB_COLLECTION_POSTS, SCRAPER_MONGODB_DB, SCRAPER_MONGODB_URL
from urllib.parse import urlparse
from datetime import timedelta

    



def get_post_collection():
    print("Connecting to MongoDB...", SCRAPER_MONGODB_URL)
    client = MongoClient(SCRAPER_MONGODB_URL)
    db = client[SCRAPER_MONGODB_DB]
    collection = db["posts"]
    return collection



def scraper_posts_cursor(platform, offset, limit, media_type=None):
    collection = get_post_collection()
    query = {}
    if platform is not None:        
        query["platform"] = { "$eq": platform }

    if media_type is not None:
        if media_type not in ["photo", "video", "text"]:
            raise ValueError("Media type must be one of: photo, video, text")
        query["media_type"] = { "$eq": media_type }
        
    posts_cursor = collection.find(query).sort("date_created").skip(offset).limit(limit)
    for post in posts_cursor:
        if platform == 'instagram':
            yield IgPost(**post)
        if platform == 'twitter':
            yield TwPost(**post)


def get_url_filename(url):
    # Parse the URL
    parsed_url = urlparse(url)
    # Extract the path component
    path = parsed_url.path
    filename = path.split('/')[-1]
    return filename





async def extract_scraper_posts(platform, partition):
    if platform not in ["twitter", "instagram", "facebook"]:
        raise ValueError("Platform must be one of: twitter, instagram, facebook")
    if type(partition) == str:
        partition = datetime.strptime(partition, "%Y-%m-%d-%H:%M")
    start_date = partition
    end_date = start_date + timedelta(hours=1)
    
    post_collection = await get_post_collection()
    filterd_posts = await post_collection.get_many({
        "date_created": {
            "$gte": start_date, 
            "$lt": end_date
        },
        "platform": {"$eq": platform}
    })
    return filterd_posts




def get_instagram_post_media(post, channel, platform):

    url = post.display_url
    media = Image.from_url(post.display_url)
    thumbnail = media.get_thumbnail()    
    filename = get_url_filename(post.display_url)
    if post.media_type.value == 'video':
        if post.video_url is not None:
            url = post.video_url
            filename = get_url_filename(post.video_url)
            media = Video.from_url(post.video_url, filename)

    media = Media(
        media=media,
        filename=filename,
        size=media.size,
        thumbnail=thumbnail,
        source=f"${platform}_{channel}",
        date=post.date_post_created,
        # caption=post.text,
        platform_id = str(post.post_id),
        platform_type = platform,
        platform_group=channel,
        url=url
    )
    media._is_downloaded = True
    return media



def scraper_instagram_post_to_news_event(post, platform) -> NewsEvent:
    channel = post.owner_username
    media = get_instagram_post_media(post, channel, platform)

    event = NewsEvent(
        date=post.date_post_created,
        text=post.text,
        media=media,
        platform_id=str(post.post_id),
        platform=platform,
        channel=channel,
        metrics=post.interaction_metrics.dict()
    )

    return event



def  get_twitter_post_media(post, channel, platform):
     media_list = []
     if post.media_type.value == 'photo':
        for image_url in post.photo_urls_list:
            image = Image.from_url(image_url)
            thumbnail = image.get_thumbnail()
            filename = get_url_filename(image_url)
            media= Media(
                media=image,
                filename=filename,
                # size=image.size,
                thumbnail=thumbnail,
                source=f"${platform}_{channel}",
                date=post.date_post_created,
                # caption=post.full_text,
                platform_id = str(post.post_id),
                platform_type = platform,
                platform_group=channel,
                url=image_url
            )
            media._is_downloaded = True
            media_list.append(media)
        for video_url in post.video_urls_list:
            video = Video.from_url(video_url)
            thumbnail = video.get_frame(0)
            filename = get_url_filename(video_url)
            media = Media(
                media=video,
                filename=filename,
                size=media.size,
                thumbnail=thumbnail,
                source=f"${platform}_{channel}",
                date=post.date_post_created,
                # caption=post.caption,
                platform_id = str(post.post_id),
                platform_type = platform,
                platform_group=channel,
                url=video_url
            )
            media._is_downloaded = True
            media_list.append(media)        
        return media_list
        


def scraper_twitter_post_to_news_event(post, platform) -> NewsEvent:
    channel = post.owner_username
    media = get_twitter_post_media(post, channel, platform)

    event = NewsEvent(
        date=post.date_post_created,
        text=post.text,
        media=media,
        platform_id=str(post.post_id),
        platform=platform,
        channel=channel,
        metrics=post.interaction_metrics.dict()
    )

    return event