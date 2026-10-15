(() => {
  const banner=document.createElement('a');banner.href='/';banner.className='live-status';banner.textContent='← Recording room · Waiting for live conversations';document.body.append(banner);
  let version='';
  async function refresh(){
    try{
      const response=await fetch('/api/map',{cache:'no-store'});if(!response.ok)throw Error('offline');const payload=await response.json();
      if(payload.points?.length){
        banner.textContent=`← Recording room · ${payload.points.length} recorded turns · ${payload.settings?.method==='pca_preview'?'Early PCA preview, not UMAP':'Live map'}`;
        if(version!==payload.created_utc){version=payload.created_utc;window.dispatchEvent(new CustomEvent('softrefusal:recordings',{detail:payload}));}
      }else banner.textContent='← Recording room · No completed turns yet · Showing an illustrative field';
    }catch{banner.textContent='← Recording room · Connection interrupted · Showing last available map';}
    setTimeout(refresh,document.hidden?15000:4000);
  }
  refresh();
})();
