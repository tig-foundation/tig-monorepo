use 
{
    crate::
    {
        ctx::
        {
            Context,
            ContextResult,
        },
    },
    std::
    {
        marker::
        {
            PhantomData,
        },
        collections::
        {
            HashMap,
        },
    },
    tig_structs::
    {
        *,
        core::
        {
            *
        }
    }
};

pub struct BlockCache<T: Context>
{
    blocks:                     HashMap<i64, Block>,
    curr_block_height:          u64,
    ctx:                        PhantomData<T>,
}

impl<T: Context> BlockCache<T>
{
    pub fn new() -> Self
    {
        return Self
        { 
            blocks:                         HashMap::new(), 
            curr_block_height:              0, 
            ctx:                            PhantomData
        };
    }

    pub async fn update_block_height(
        &mut self
    )                                   -> ContextResult<()>
    {
        return Ok(());
    }

    pub async fn fetch_latest_blocks(
        &mut self, 
        ctx:                    &T
    )                                   -> ContextResult<()>
    {
        let latest_block                    = ctx.get_block_by_height(-1, false).await?;
        let latest_block_height             = latest_block.unwrap().details.height as u64;

        if latest_block_height > self.curr_block_height
        {
            for height in (self.curr_block_height + 1)..=latest_block_height
            {
                self.fetch_block(ctx, height as i64).await.expect("");
            }
        }

        return Ok(());
    }

    pub async fn fetch_block(
        &mut self,
        ctx:                    &T,
        mut height:             i64
    )                                   -> Option<&Block>
    {
        if height < 0
        {
            let _                           = self.update_block_height().await;

            height                          = self.curr_block_height as i64 + height + 1;
        }

        if height < 0 || height > self.curr_block_height as i64
        {
            return None;
        }

        if !self.blocks.contains_key(&height)
        {
            let block                       = ctx.get_block_by_height(height as i64, false).await.ok()?;
            self.blocks.insert(height, block.expect("Block not found"));
        }

        return self.blocks.get(&height);
    }

    pub async fn get_block(
        &self,
        ctx:                    &T,
        mut height:             i64
    )                                   -> Option<&Block>
    {
        if height < 0
        {
            height                          = self.curr_block_height as i64 + height + 1;
        }

        if height < 0 || height as u64 > self.curr_block_height
        {
            return None;
        }

        return self.blocks.get(&height);
    }

    pub async fn ensure_has_latest_blocks(
        &mut self,
        ctx:                    &T
    )                                   -> ContextResult<()>
    {
        return self.fetch_latest_blocks(ctx).await;
    }
}