from enum import Enum
from typing import Any, List, Union
from urllib import response
from pydantic import BaseModel, Field, validator
import requests
from components.text.transcript import Transcript

from config import BACKEND_URL, BACKEND_PASSWORD, BACKEND_USER, BACKEND_JWT
from requests.auth import HTTPBasicAuth
from datetime import datetime
import asyncio

try:
    from components.video.video_shots import Shot
except:
    Shot = Any

def pack_args(**args):
    return {k: v for k, v in args.items() if v is not None}


class MediaType(Enum):
    IMAGE = 'IMAGE'
    VIDEO = 'VIDEO'
    AUDIO = 'AUDIO'
    TEXT = 'TEXT'
    UNKNOWN = 'UNKNOWN'





# class MediaBERec(BaseModel):
#     id: str
#     create_at: datetime = Field(alias="createAt")
#     update_at: datetime = Field(alias="updateAt")
#     type: MediaType
#     name: str
#     source: str = None
#     title: str = None
#     url: str = None
#     thumbnail: str
#     height: int
#     width: int
#     video_duration: float = Field(alias="videoDuration")
#     fps: int = None
#     audio: str = None
#     transcript_tags: Any = Field(alias="transcriptTags")
#     transcript: Any = None
#     creator: str = None
#     license: str = None
#     tags: Any = None
#     source_url: str = Field(alias="sourceUrl")
#     isProcessed: bool 
#     error: str = None
#     faces: Any = None
#     events: Any = None
#     metadata: Any = None
class MediaBERec(BaseModel):
    id: str
    create_at: datetime = Field(alias="createAt")
    update_at: datetime = Field(alias="updateAt")
    type: MediaType
    name: str
    title: Union[str, None] = None
    url: Union[str, None] = None
    thumbnail: str
    height: int
    width: int
    video_duration: Union[float, None] = Field(alias="videoDuration", default=None)
    source: Union[str, None] = None
    creator: Union[str, None] = None
    license: Union[str, None] = None
    is_rocessed: bool = Field(alias="isProcessed")
    faces: Any
    events: List[Shot]

    @validator('create_at', pre=True)
    def parse_create_at(cls, v):
        return datetime.fromtimestamp(int(v) / 1000)
    
    @validator('update_at', pre=True)
    def parse_update_at(cls, v):
        return datetime.fromtimestamp(int(v) / 1000)
    
    @validator('events', pre=True)
    def parse_events(cls, v):
        return [Shot.from_json(event) for event in v]
    
    class Config:
        arbitrary_types_allowed = True


    # class Config:
    #     arbitrary_types_allowed = True



class Backend:


    def __init__(self, backend_url=BACKEND_URL, backend_jwt=BACKEND_JWT, backend_user=BACKEND_USER, backend_password=BACKEND_PASSWORD):
        self.backend_url = backend_url
        self.backend_jwt = backend_jwt
        self.backend_user = backend_user
        self.backend_password = backend_password
        


    def get_movie(self, account_id: str,movie_id):
        if not movie_id:
            raise Exception('Movie id is required')
        query = '''
        query GetMovie($id: String!) {
            movie(id: $id) {
                id
                title
                description
                background
                screens {
                    id
                    isTitle
                    title
                    text
                    originalText
                    content
                    voice
                    voiceId
                    voiceOffset
                    duration
                    startAt
                    presignedUrl
                    transcriptTag
                    metadata
                    attachedMedia {
                        id
                        rank                    
                        
                        media {                    
                            id
                            type
                            url
                            thumbnail
                            name
                            fps
                            creator
                            license
                            tags
                            isProcessed
                            faces
                            events
                            source
                            height
                            width                                                
                        }
                        
                        videoStartAt
                        videoDuration

                        lane

                        startFrom
                        duration
                        animations
                    }
                }
                music {
                    id
                    name
                    duration
                    url
                }
            }
        }
        '''
        response = self.post_graphql_query(query, variables = {
            'id': movie_id
        }, account_id = account_id)
        return response['data']['movie']


    def create_movie(self, account_id: str,title, description=None):
        query = '''
            mutation CreateMovie($title: String!, $description: String) {
                createMovie(title: $title, description: $description) {
                    id
                    title
                }
            }
        '''
        response = self.post_graphql_query(query, variables = {
            'title': title,
            'description': description
        }, account_id = account_id)
        return response['data']['createMovie']



    def edit_movie(self, account_id: str,movie_id, title, description = None):
        query = '''
            mutation EditMovie($id: String!, $title: String!, $description: String) {
                editMovie(id: $id, title: $title, description: $description) {
                    id
                    title
                    description
                }
            }
        '''
        response = self.post_graphql_query(query, variables = {
            'id': movie_id,
            'title': title,
            'description': description
        }, account_id=account_id)
        return response['data']['editMovie']


    def add_media(
            self,
            name: str, 
            type: str, 
            source: str, 
            video_duration: float = None, 
            width: int = None, 
            height: int = None, 
            is_processed: bool = None,
            account_id: str = None, 
            movie_context: str = None,
            url: str = None,
            thumbnail: str = None,
            faces=None, 
            metadata=None,
            events=None,
            media_id=None,
        ):
        query = '''
            mutation AddMedia($name: String!, $type: String!, $source: String!, $height: Int , $width: Int, $videoDuration: Float, $faces: Json, $events: Json, $isProcessed: Boolean, $url: String, $thumbnail: String, $mediaId: String, $metadata: Json, $movieContext: String!) {
                addMedia(name: $name, type: $type, source: $source, height: $height, width: $width, videoDuration: $videoDuration, faces: $faces, events: $events, isProcessed: $isProcessed, url: $url, thumbnail: $thumbnail, mediaId: $mediaId, metadata: $metadata, movieContext: $movieContext) {
                    id
                    name
                    type
                    source
                    height
                    width
                    videoDuration
                    isProcessed
                }
            }
        '''
        variables = pack_args(
            name = name,
            type = type,
            source = source,
            height = height,
            width = width,
            isProcessed = is_processed,
            videoDuration = video_duration,
            movieContext = movie_context,
            url = url,
            thumbnail = thumbnail,
            faces = faces,
            events = events,
            metadata=metadata,
            mediaId=media_id,            
        )
        response = self.post_graphql_query(
            query, 
            variables = variables, 
            account_id=account_id
        )
        # response = self.post_graphql_query(query, variables = {
        #     'name': name,
        #     'type': type,
        #     'source': source,
        #     'height': height,
        #     'width': width,
        #     'isProcessed': is_processed,
        #     'videoDuration': video_duration,
        #     'movieContext': movie_context,
        #     'url': url,
        #     'thumbnail': thumbnail,
        #     'faces': faces,
        #     'events': events
        # }, account_id=account_id)
        return response['data']['addMedia']




    def add_multi_media(self, account_id, media):
        query = '''
            mutation AddMultiMedia($media: [MediaArgs!]!) {
                addMultiMedia(media: $media) {
                    id
                    name
                    type
                    source
                    height
                    width
                    videoDuration
                    isProcessed
                }
            }
        '''
        response = self.post_graphql_query(query, variables = {
            'media': media
        }, account_id=account_id)
        return response['data']['addMultiMedia']



    def get_thumbnail_media(self, source='hasbara'):
        query = '''
            query GetMediaThumbnails($source: String!, $movieContext: String!) {
                mediaThumbnails(source: $source, movieContext: $movieContext) {
                    id
                    createAt
                    updateAt
                    type
                    name
                    source
                    title
                    url
                    thumbnail
                    height
                    width

                    videoDuration
                    fps
                    audio
                    audioPresignedUrl

                    transcriptTags
                    transcript
                    
                    creator
                    sourceUrl
                    license
                    tags
                    
                    faces
                    events

                    isProcessed
                    error
                    
                    metadata
                    movieContext
                }
                getMediaSources(movieContext: $movieContext) {
                    source
                }
            }
        '''
        variables = pack_args(
            source=source,
            movieContext='-1'
        )
        response = self.post_graphql_query(query, variables=variables)
        return response['data']['mediaThumbnails']


    def get_media(self, media_id: str):
        query = '''
            query GetMedia($id: String!) {
                getMedia(id: $id) {
                    id
                    createAt
                    updateAt
                    type
                    name
                    title
                    url
                    thumbnail
                    height
                    width
                    videoDuration
                    source
                    creator
                    license
                    isProcessed
                    faces
                    events
                }
            }
        '''
        response = self.post_graphql_query(query, variables = {
            'id': media_id
        })
        media_dict = response['data']['getMedia']
        return media_dict
        # return MediaBERec(**media_dict)
    
    async def aget_media(self, media_id: str) -> MediaBERec:
        query = '''
            query GetMedia($id: String!) {
                getMedia(id: $id) {
                    id
                    createAt
                    updateAt
                    type
                    name
                    title
                    url
                    thumbnail
                    height
                    width
                    videoDuration
                    source
                    creator
                    license
                    isProcessed
                    faces
                    events
                }
            }
        '''
        response = await asyncio.to_thread(
            self.post_graphql_query,
            query=query,
            variables = {
                'id': media_id
            })
        media_dict = response['data']['getMedia']
        if media_dict is None:
            return None
        # return media_dict
        try:
            return MediaBERec(**media_dict)
        except Exception as e:
            print(media_dict)
            raise e
        


    def delete_media(self, media_id: str):
        query = '''
        mutation DeleteMedia($id: String!) {
            deleteMedia(id: $id) {
                id
            }
        }    
        '''
        response = self.post_graphql_query(query, variables = {
            'id': media_id
        })
        return response['data']['deleteMedia']


    def edit_media(
        self,
        media_id: str, 
        name: str = None, 
        type: str = None, 
        source: str = None, 
        video_duration: float = None, 
        width: int = None, 
        height: int = None, 
        fps: int = None,
        audio: str = None,
        transcript_tags: bool = None,
        thumbnail: str = None,
        faces=[], 
        events=[],
        is_processed: bool = None, 
        error: bool = None,
        account_id: str = None,    
        ):
        query = '''
            mutation EditMedia($id: String!, $name: String, $type: String, $source: String, $height: Int , $width: Int, $videoDuration: Float, $fps: Int,  $audio: String, $transcriptTags: [String] ,$thumbnail: String, $faces: Json, $events: Json, $isProcessed: Boolean, $error: String) {
                editMedia(id: $id, name: $name, type: $type, source: $source, height: $height, width: $width, fps: $fps, videoDuration: $videoDuration, audio: $audio, transcriptTags: $transcriptTags, thumbnail: $thumbnail, faces: $faces, events: $events, isProcessed: $isProcessed, error: $error) {
                    id
                    name
                    type
                    source
                    height
                    width
                    videoDuration
                    faces
                    isProcessed
                    error  
                }
            }
        '''

        variables = pack_args(
            id = media_id,
            name = name,
            type = type,
            source = source,
            height = height,
            width = width,
            fps=fps,
            videoDuration = video_duration,
            faces = faces,
            events = events,
            isProcessed = is_processed,
            error = error,
            audio = audio,
            transcriptTags=transcript_tags,
            thumbnail = thumbnail
        )
        response = self.post_graphql_query(query, variables=variables)
        return response


    def delete_all_media(self, source: str=None):
        query = '''
            mutation DeleteAllMedia($source: String) {
                deleteAllMedia(source: $source) {
                    id
                    name
                    thumbnail
                }
            }
        '''
        response = self.post_graphql_query(query, variables = {
            'source': source
        })
        return response['data']['deleteAllMedia']

    def post_music(self, name: str, filename: str, duration: float):
        query = '''
            mutation AddMusic($name: String!, $filename: String! $duration: Float!) {
                addMusic(name: $name, filename: $filename, duration: $duration) {
                    id
                    name
                    filename
                    duration
                }
            }
        '''
        response = self.post_graphql_query(query, variables = {
            'name': name,
            'duration': duration,
            'filename': filename
        })
        return response


    def get_screen(self, screen_id):
        query = '''
            query GetScreen($id: String!) {
                screen(id: $id) {
                    id
                    title
                    text
                    voice
                    duration
                    content
                    metadata
                }
            }
        '''
        response = self.post_graphql_query(query, variables = {
            'id': screen_id
        })
        return response['data']['screen']


    def get_screen_transcript(self, screen_id):
        screen = self.get_screen(screen_id)
        transcript = Transcript.from_lexical(screen['content'])
        return transcript


    def add_screen(self, account_id: str, movie_id, is_title=None, text=None, voice=None, duration=None, after_screen_id=None):
        if movie_id is None:
            raise Exception('Movie id is required')
        # if text == None and voice == None and duration == None:
            # raise Exception('At least one parameter is required')
        query = '''
            mutation AddScreen($movieId: String!, $isTitle: Boolean, $title: String, $text: String, $voice: String, $duration: Float, $afterScreenId: String) {
                addScreen(movieId: $movieId, isTitle: $isTitle, title: $title, text: $text, voice: $voice, duration: $duration, afterScreenId: $afterScreenId) {
                    id
                    title
                    text
                    voice
                    duration
                    presignedUrl
                }
            }
        '''

        variables = pack_args(
            movieId= movie_id,
            isTitle= is_title,
            text= text,
            voice= voice,
            duration= duration,
            afterScreenId= after_screen_id
        )

        response = self.post_graphql_query(query, variables = variables, account_id = account_id)
        return response['data']['addScreen']

    def delete_screen(self, account_id: str, screen_id):
        if not screen_id:
            raise Exception('Screen id is required')
        query = '''
            mutation DeleteScreen($id: String!) {
                deleteScreen(id: $id) {
                    id
                }
            }
            '''
        res = self.post_graphql_query(query, variables = {
                "id": screen_id
            }, account_id = account_id)
        return res



    def update_screen(self, account_id: str, screen_id=None, text=None, is_title=None, content=None, duration=None, transcript_tags=None):
        query = '''
            mutation EditScreen($id: String!, $isTitle: Boolean, $title: String, $text: String, $content: Json, $voiceText: String, $voice: String, $voiceId: String, $duration: Float, $transcriptTag: String) {
                editScreen(id: $id, isTitle: $isTitle, title: $title, text: $text, content: $content, voiceText: $voiceText, voice: $voice, voiceId: $voiceId, duration: $duration, transcriptTag: $transcriptTag) {
                    id
                    title
                    text
                    content
                    isTitle
                    voice
                    voiceId
                    duration
                    transcriptTag
                }
            }
        '''
        variables = pack_args(
            id = screen_id,
            text = text,
            is_title = is_title,
            content = content,
            duration = duration,
            transcriptTag = transcript_tags
        )
        response = self.post_graphql_query(query, variables = variables, account_id=account_id)
        return response


    def update_multiple_screens(self, account_id: str, screens):
        query = '''
            mutation EditMultipleScreens($screens: [EditScreenInput!]!) {
                editMultipleScreens(screens: $screens) {
                    ids          
                }
            }
        '''
        screen_data = []
        for screen in screens:
            screen_data.append(pack_args(
                id = screen['id'],
                text = screen.get('text', None),
                isTitle = screen.get('is_title', None),
                content = screen.get('content', None),
                duration = screen.get('duration', None),
                transcriptTag = screen.get('transcriptTags', None),
                voiceText = screen.get('voiceText', None),
                voice = screen.get('voice', None),
                voiceId = screen.get('voiceId', None),
                originalText = screen.get('originalText', None)
            ))
        variables = {
            'screens': screen_data
        }
        response = self.post_graphql_query(query, variables = variables, account_id=account_id)
        return response


    def update_screen_url(self, account_id: str, screen_id, voice_id=None, voice_text=None, voice=None, duration=None, transcript_tags=None, transcript=None, content=None):
        query = '''
            mutation EditScreen($id: String!, $isTitle: Boolean, $title: String, $text: String, $voiceText: String, $voice: String, $voiceId: String, $duration: Float, $transcriptTag: String, $transcript: Json, $content: Json) {
                editScreen(id: $id, isTitle: $isTitle, title: $title, text: $text, voiceText: $voiceText, voice: $voice, voiceId: $voiceId, duration: $duration, transcriptTag: $transcriptTag, transcript: $transcript, content: $content) {
                    id
                    title
                    text
                    voice
                    voiceId
                    duration
                    transcriptTag
                    transcript
                    content
                }
            }
        '''
        variables = pack_args(
            id = screen_id,
            voiceText = voice_text,
            voice = voice,
            voiceId = voice_id,
            duration = duration,
            transcriptTag = transcript_tags,
            transcript = transcript,
            content = content
        )
        response = self.post_graphql_query(query, variables = variables, account_id=account_id)
        return response


    def add_media_to_screen(self, account_id: str,screen_id, media_id):
        query = '''
            mutation AddMediaToScreen($screenId: String!, $mediaId: String!) {
                addMediaToScreen(screenId: $screenId, mediaId: $mediaId) {
                    id
                    attachedMedia {
                        id
                    }
                }
            }
        '''
        response = self.post_graphql_query(query, variables = {
            'screenId': screen_id,
            'mediaId': media_id
        }, account_id=account_id)
        return response['data']['addMediaToScreen']


    def add_multi_media_to_screen(self, account_id: str,screen_id, attached_media, delete_previous=False):
        query = '''
            mutation AddMultiMediaToScreen($screenId: String!, $media: [AttachedMediaInput!]!, $deletePrevious: Boolean) {
                addMultiMediaToScreen(screenId: $screenId, media: $media, deletePrevious: $deletePrevious) {
                    id
                    attachedMedia {
                        id
                    }
                }
            }
        '''
        response = self.post_graphql_query(query, variables = {
            'screenId': screen_id,
            'media': attached_media,
            'deletePrevious': delete_previous
        }, account_id=account_id)
        return response['data']['addMultiMediaToScreen']

    def update_render(self, account_id: str,renderId, jobId=None, status=None, progress=None,url=None):   
        query = '''
            mutation UpdateRender($renderId: String!, $jobId: String, $status: String, $progress: Float, $url: String) {
            updateRender(renderId: $renderId, jobId: $jobId, status: $status, progress: $progress, url: $url) {
                id
                jobId
                status
                progress
                url
            }
        }
        '''
        variables = {}
        variables['renderId'] = renderId
        if jobId:
            variables['jobId'] = jobId
        if status:
            variables['status'] = status
        if progress:
            variables['progress'] = progress
        if url:
            variables['url'] = url
        response = self.post_graphql_query(query, variables = variables, account_id=account_id)
        return response


    def update_scrape(self, account_id: str,scrape_id, jobId=None, status=None, progress=None,url=None):   
        query = '''
            mutation UpdateScrape($id: String!, $jobId: String, $status: String) {
                updateScrape(id: $id, jobId: $jobId, status: $status) {
                    id
                    jobId
                    status
                    url
                }
        }
        '''
        variables = {}
        variables['id'] = scrape_id
        if jobId:
            variables['jobId'] = jobId
        if status:
            variables['status'] = status
        response = self.post_graphql_query(query, variables = variables, account_id=account_id)
        return response['data']['updateScrape']



    def update_prompt(self, account_id: str, message_id, input=None, output=None, error=None, status=None, command_type=None, media=None):
        query = '''
            mutation UpdatePrompt($id: String!, $input: Json, $output: Json, $error: String, $status: String, $commandType: String, $media: [MediaArgs!]) {
                updatePrompt(id: $id, input: $input, output: $output, error: $error, status: $status, commandType: $commandType, media: $media) {
                    id
                }
            }
        '''
        variables = pack_args(
            id = message_id,
            input = input,
            output = output,
            error = error,
            status = status,
            commandType = command_type,
            media = media
        )
        response = self.post_graphql_query(query, variables = variables, account_id=account_id)
        return response



    NEWS_EVENTS_FIELDS = """
                    id
                    createAt
                    updateAt
                    data
                    text
                    facts
                    opinions
                    date
                    platform
                    channel
                    type
                    tags
                    reliability
                    url
                    media
    """

    def get_news_events(self, ids=None, offset=0, limit=10, from_date=None, to_date=None, search=None):
        query = """
            query GetNewsEvents($ids: [String!], $offset: Int!, $limit: Int!, $fromDate: String, $toDate: String, $search: String) {
                getNewsEvents(ids: $ids, offset: $offset, limit: $limit, fromDate: $fromDate, toDate: $toDate, search: $search) {
                    id
                    createAt
                    updateAt
                    data
                    text
                    facts
                    opinions
                    date
                    platform
                    channel
                    type
                    tags
                    reliability
                    url
                    topic
                    media {
                        id
                        name
                        thumbnail
                        type
                        height
                        width
                        videoDuration
                        fps                    
                    }
                }
            }
    """
        variables = pack_args(
            ids = ids,
            offset = offset,
            limit = limit,
            fromDate = from_date,
            toDate = to_date,
            search = search        
        )
        response = self.post_graphql_query(query, variables = variables)
        return response['data']['getNewsEvents']


    def get_news_event_by_id(self, id):
        query = """
            query GetNewsEventById($id: String!){
                getNewsEventById(id: $id) {
                    id
                    createAt
                    updateAt
                    data
                    text
                    facts
                    opinions
                    date
                    platform
                    channel
                    type
                    tags
                    reliability
                    url
                    topic
                    media {
                        id
                    }
                }
            }
    """
        variables = pack_args(
            id = id
        )
        response = self.post_graphql_query(query, variables = variables, account_id='-1')
        return response['data']['getNewsEventById']

    def create_news_event(
            self,
            id,
            data, 
            text, 
            facts, 
            opinions, 
            date, 
            platform, 
            channel, 
            platform_id, 
            type, 
            tags, 
            reliability, 
            url, 
            media_ids,
            metrics,
            topic=None,
            account_id=None,
            movie_id=None,
        ):
        mutation = """
    mutation CreateNewsEvent($id:String, $data: Json, $text: String, $facts: Json, $opinions: Json, $date: String, $platform: String!, $channel: String!, $platformId: String!, $type: String, $tags: Json, $reliability: Float, $url: String, $mediaIds: [String], $metrics: Json, $topic: String, $movieId: String, $accountId: String) {
        createNewsEvent(id: $id, data: $data, text: $text, facts: $facts, opinions: $opinions, date: $date, platform: $platform, channel: $channel, platformId: $platformId, type: $type, tags: $tags, reliability: $reliability, url: $url, mediaIds: $mediaIds, metrics: $metrics, topic: $topic, movieId: $movieId, accountId: $accountId) {
            id
            createAt
            updateAt
        }
    }
    """
        variables = pack_args(
            id = id,
            data = data,
            text = text,
            facts = facts,
            opinions = opinions,
            date = date,
            platform = platform,
            channel = channel,
            platformId = platform_id,
            type = type,
            tags = tags,
            reliability = reliability,
            url = url,
            mediaIds = media_ids,
            metrics=metrics,
            topic=topic,
            movieId=movie_id,
            accountId=account_id
        )
        response = self.post_graphql_query(mutation, variables = variables, account_id='-1')
        return response['data']['createNewsEvent']

    def update_news_event(
            self,
            id, 
            data=None, 
            text=None, 
            facts=None, 
            opinions=None, 
            date=None, 
            platform=None, 
            channel=None,
            platform_id=None, 
            type=None, 
            tags=None, 
            reliability=None, 
            url=None, 
            media_id=None,
            metrics=None,
            topic=None
        ):
        mutation = """
    mutation UpdateNewsEvent($id: String!, $data: Json, $text: String, $facts: Json, $opinions: Json, $date: String, $platform: String, $channel: String, $platformId: String, $type: String, $tags: Json, $reliability: Float, $url: String, $mediaId: [String], $metrics: Json, $topic: String) {
        updateNewsEvent(id: $id, data: $data, text: $text, facts: $facts, opinions: $opinions, date: $date, platform: $platform, channel: $channel, platformId: $platformId, type: $type, tags: $tags, reliability: $reliability, url: $url, mediaId: $mediaId, metrics: $metrics, topic: $topic) {
            id
            createAt
            updateAt
        }
    }
    """
        variables = pack_args(
            id = id,
            data = data,
            text = text,
            facts = facts,
            opinions = opinions,
            date = date,
            platform = platform,
            channel = channel,
            platformId = platform_id,
            type = type,
            tags = tags,
            reliability = reliability,
            url = url,
            mediaId = media_id,
            metrics=metrics,
            topic=topic
        )
        response = self.post_graphql_query(mutation, variables = variables, account_id="-1")
        return response['data']['updateNewsEvent']


    def delete_news_event(self, id):
        mutation = """
            mutation DeleteNewsEvent($id: String!) {
                deleteNewsEvent(id: $id) {
                    id
                }
            }
        """
        variables = {
            'id': id
        }
        response = self.post_graphql_query(mutation, variables = variables, account_id="-1")
        return response['data']['deleteNewsEvent']


    def delete_all_news_events(self):
        mutation = """
            mutation DeleteAllNewsEvents {
                deleteAllNewsEvents {
                    id
                }
            }
        """
        response = self.post_graphql_query(mutation, account_id="-1")
        return response['data']['deleteAllNewsEvents']


    def post_graphql_query(self, query, variables=None, account_id=None):
        url = f'{BACKEND_URL}/api/graphql'
        response = requests.post(
            url, 
            json={
                'query': query,
                'variables': variables
            },
            auth=HTTPBasicAuth(BACKEND_USER, BACKEND_PASSWORD),
            headers={'backend-auth': BACKEND_JWT, 'account-id': account_id},
            timeout=5,
            verify=False
        )
        j = response.json()
        if response.status_code != 200 or j.get("errors", None) is not None:
            if j:
                raise Exception(f'Error: {response.status_code} {j.get("errors", response.text)}')
            raise Exception(f'Error: {response.status_code}')
        return response.json()