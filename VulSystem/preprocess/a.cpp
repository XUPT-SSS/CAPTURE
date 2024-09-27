already_AddRefed<nsPIWindowRoot> nsGlobalWindow::GetTopWindowRoot()
{
  nsPIDOMWindow* piWin = GetPrivateRoot();
    if (!piWin) 
	{
	    return nullptr;
	}
		  
	nsCOMPtr<nsPIWindowRoot> window = do_QueryInterface(piWin->GetChromeEventHandler());
	return window.forget();
}